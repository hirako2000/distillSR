The checkpoint system maintains two distinct file formats to balance training resumability against deployment convenience. Full checkpoints contain complete training state (.pt) while weight-only exports provide lightweight files for inference (.pth). 

Produced in the [./weights] folder.

## Dual format

Full checkpoints saved with the .pt extension contain the model state dictionary, optimizer state, scheduler state, current iteration and epoch counts, best validation metrics, and complete training configuration. These files enable exact training resumption from any point, preserving all optimizer statistics and learning rate scheduler progress.

Weight-only exports use the .pth extension and contain only the model parameters in a dictionary with a single params key. This format matches the structure of pretrained weights from external sources and provides minimal files for deployment. The size difference is substantial, with full checkpoints approximately three times larger than weight-only exports.

## Checkpoints

During regular checkpoint intervals, both formats are saved. The full checkpoint uses the pattern `experiment_name_iter_N.pt` while the weight-only export uses `experiment_name_iter_N.pth`. When a new best model is discovered during validation, both formats are saved with best suffixes, and the logger additionally archives a copy in the experiment log directory.


## Pretrained Weights

The same loading logic handles pretrained weights from external sources. When a pretrain_path is specified in configuration, the trainer loads the weights using the checkpoint manager before training begins. This supports both the weight-only format used by pretrained models and the full checkpoint format used for resuming interrupted training runs.

## Resume

Resuming training needs the full checkpoint format containing optimizer and scheduler states. The resume_path configuration points to a .pt file, which the trainer loads to restore model weights, optimizer statistics, scheduler progress, iteration count, epoch count, and best metric values. This enables training to continue exactly where it left off, essential for long-running experiments.