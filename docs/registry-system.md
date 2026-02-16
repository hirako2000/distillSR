
Registry provides dynamic model instantiation through a decorator-based registration system. This design also used for blocks, and upsamplers so that they can be added without modifying the training code, and enables configuration-driven model selection through YAML files.

## Structure

Three separate registries manage different component types. The model registry stores complete architectures accessible by name. The block registry contains reusable components like DCCM, EA, and PLKConv2d that can be composed into larger networks. The upsampler registry holds upsampling modules including DySample and PixelShuffle for flexible decoder configuration.

## Decorators

Components register themselves using decorators that accept a name identifier. When a class is decorated with register_model, register_block, or register_upsampler, it is automatically added to the corresponding registry at module import time. This ensures all available components are discoverable without manual enumeration.

## Dynamic instantiation

The get_model, get_block, and get_upsampler functions retrieve registered components by name and instantiate them with provided keyword arguments. This allows configuration files to specify components as strings, with the registry system handling the actual construction. The factory module provides a clean public interface, hiding registry internals from calling code.

## Configuration

The training configuration accepts component names as strings, which are resolved through the registry during model construction. This enables experiments to swap architectures or upsamplers simply by changing configuration values rather than modifying code. The same registry system supports both the main RealPLKSR architecture and any future variants added to the codebase.

## Benefits

During development, faced mutliple obstacles trying to support MPS with optmisations that pytorch would not support (yet). For example, the Pixel shuffling technique can easily be implemented on MPS, but dysampling is so inneficient it was not an option for non CUDA drivers.

This approach decouples component definition from instantiation logic, making the codebase more maintainable and extensible. New architectures can be added by simply creating the module and applying the appropriate decorator. The registry also enables programmatic discovery of available components for CLI help text and validation.

For details on how specific architectures are constructed using registered components, refer to [architecture-details](./architecture-details.md).