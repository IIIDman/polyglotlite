"""Training loop for PolyglotLite models"""

import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from polyglotlite.training.config import TrainingConfig


class TextDataset(Dataset):
    """Simple dataset that tokenizes text on the fly"""
    
    def __init__(
        self, 
        texts: List[str],
        tokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text, max_length=self.max_length)
        
        # Pad or truncate
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        
        input_ids = torch.tensor(tokens[:-1])
        labels = torch.tensor(tokens[1:])
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class Trainer:
    """
    Trainer for PolyglotLite models.
    
    Example:
        >>> model = PolyglotLite.from_pretrained("polyglot-135m")
        >>> trainer = Trainer(model=model, train_data="data.json")
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model,
        train_data: Optional[Union[str, List[str], Dataset]] = None,
        eval_data: Optional[Union[str, List[str], Dataset]] = None,
        config: Optional[TrainingConfig] = None,
        use_qlora: bool = False,
        learning_rate: float = 5e-4,
        batch_size: int = 8,
        max_steps: int = 10000,
        output_dir: str = "./outputs",
        **kwargs
    ):
        """
        Initialize trainer.
        
        Args:
            model: PolyglotLite model to train
            train_data: Training data (file path, list of texts, or Dataset)
            eval_data: Evaluation data
            config: Training configuration
            use_qlora: Use QLoRA for efficient fine-tuning
            learning_rate: Learning rate
            batch_size: Batch size
            max_steps: Maximum training steps
            output_dir: Directory for outputs
        """
        self.model = model
        self.config = config or TrainingConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_steps=max_steps,
            output_dir=output_dir,
            use_qlora=use_qlora,
            **kwargs
        )
        
        # Setup device
        self.device = self._setup_device()
        self.model.to(self.device)
        
        # Setup data
        self.train_dataset = self._setup_dataset(train_data)
        self.eval_dataset = self._setup_dataset(eval_data) if eval_data else None
        
        # Setup training components
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler() if self.config.use_amp else None
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Setup output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_device(self) -> str:
        """Setup and return device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.config.device
    
    def _setup_dataset(
        self, 
        data: Optional[Union[str, List[str], Dataset]]
    ) -> Optional[Dataset]:
        """Setup dataset from various input formats."""
        if data is None:
            return None
        
        if isinstance(data, Dataset):
            return data
        
        # Load tokenizer if needed
        if self.model.tokenizer is None:
            self.model.load_tokenizer()
        
        if isinstance(data, str):
            # Load from file
            path = Path(data)
            if path.suffix == '.json':
                with open(path) as f:
                    texts = json.load(f)
                if isinstance(texts, dict):
                    texts = texts.get('texts', texts.get('data', []))
            elif path.suffix == '.txt':
                with open(path) as f:
                    texts = f.read().split('\n\n')
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        else:
            texts = data
        
        return TextDataset(
            texts=texts,
            tokenizer=self.model.tokenizer,
            max_length=self.config.max_seq_len
        )
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        # Separate parameters that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'norm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        self.optimizer = AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
        )
        
        # Setup scheduler with warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config.warmup_steps
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_steps - self.config.warmup_steps,
            eta_min=self.config.learning_rate * 0.1
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_steps]
        )
    
    def _compute_loss(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute cross-entropy loss."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        outputs = self.model.model(input_ids)
        logits = outputs["logits"]
        
        # Flatten for cross entropy
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=self.model.tokenizer.pad_token_id
        )
        
        return loss
    
    def train(self, resume_from: Optional[str] = None):
        """
        Train the model.
        
        Args:
            resume_from: Path to checkpoint to resume from
        """
        if self.train_dataset is None:
            raise ValueError("No training data provided")
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        # Create dataloader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=min(self.config.num_workers, 4),
            pin_memory=True if self.device == "cuda" else False,
        )
        
        # Training loop
        self.model.model.train()
        
        print(f"Starting training on {self.device}")
        print(f"Total steps: {self.config.max_steps}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        
        accumulated_loss = 0.0
        start_time = time.time()
        
        while self.global_step < self.config.max_steps:
            for batch in train_loader:
                # Forward pass
                if self.config.use_amp and self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        loss = self._compute_loss(batch)
                else:
                    loss = self._compute_loss(batch)
                
                loss = loss / self.config.gradient_accumulation_steps
                accumulated_loss += loss.item()
                
                # Backward pass
                if self.config.use_amp and self.device == "cuda":
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights
                if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.use_amp and self.device == "cuda":
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 1.0)
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.log_interval == 0:
                    avg_loss = accumulated_loss / self.config.log_interval
                    elapsed = time.time() - start_time
                    steps_per_sec = self.global_step / elapsed
                    
                    print(
                        f"Step {self.global_step}/{self.config.max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {self.scheduler.get_last_lr()[0]:.2e} | "
                        f"Steps/sec: {steps_per_sec:.2f}"
                    )
                    accumulated_loss = 0.0
                
                # Evaluation
                if self.global_step % self.config.eval_interval == 0 and self.eval_dataset:
                    eval_loss = self.evaluate()
                    print(f"Eval loss: {eval_loss:.4f}")
                    
                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        self.save_checkpoint("best")
                    
                    self.model.model.train()
                
                # Save checkpoint
                if self.global_step % self.config.save_interval == 0:
                    self.save_checkpoint(f"step-{self.global_step}")
                
                if self.global_step >= self.config.max_steps:
                    break
        
        # Save final model
        self.save_checkpoint("final")
        print("Training complete!")
    
    def evaluate(self) -> float:
        """Evaluate the model and return average loss."""
        if self.eval_dataset is None:
            return 0.0
        
        self.model.model.eval()
        
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                loss = self._compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, name: str):
        """Save training checkpoint."""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        torch.save(state, checkpoint_dir / "training_state.pt")
        
        # Save config
        self.config.save(checkpoint_dir / "training_config.json")
        
        print(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint_dir = Path(path)
        
        # Load model
        self.model = self.model.from_pretrained(checkpoint_dir)
        self.model.to(self.device)
        
        # Load training state
        state = torch.load(checkpoint_dir / "training_state.pt")
        self.global_step = state["global_step"]
        self.best_loss = state["best_loss"]
        
        if self.optimizer:
            self.optimizer.load_state_dict(state["optimizer_state"])
        if self.scheduler:
            self.scheduler.load_state_dict(state["scheduler_state"])
        
        print(f"Checkpoint loaded from {checkpoint_dir}")
