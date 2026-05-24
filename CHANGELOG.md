# Changelog

causeinfer tries to follow [semantic versioning](https://semver.org/), a MAJOR.MINOR.PATCH version where increments are made of the:

- MAJOR version when we make incompatible API changes
- MINOR version when we add functionality in a backwards compatible manner
- PATCH version when we make backwards compatible bug fixes

## causeinfer 2.0.0

- All functions were typed and docstrings were expanded
- `prek` based pre-commit hooks are used to improve package development
- `Ruff` is now used for formatting and import sorting instead of `black`
- Linting is now done with `ty` instead of `mypy`
- Dependency management is now done via `uv`
- All production and development dependencies were updated
- Tests and GitHub workflows were updated given the above changes

## causeinfer 1.0.1

- Updates source code files with direct references to codes they're based on

## causeinfer 1.0.0

- Release switches causeinfer over to [semantic versioning](https://semver.org/) and indicates that it is stable

## causeinfer 0.1.2

Changes include:

- An src structure has been adopted to improve organization and testing
- Users are now able to implement the following models:
  - Reflective Uplift (Shaar 2016)
  - Pessimistic Uplift (Shaar 2016)
- The contribution guidelines have been expanded
- Code quality checks via Codacy have been added
- Extensive code formatting has been done to improve quality and style
- Bug fixes and a more explicit use of exceptions

## causeinfer 0.1.0

First stable release of causeinfer

- Users are able to implement baseline causal inference models including:
  - Two model
  - Interaction term (Lo 2002)
  - Binary transformation (Lai 2006)
  - Quaternary transformation (Kane 2014)
- Plotting functions allow for graphical analysis of models
- Functions useful for research such as model iterations, oversampling, and variance analysis are included
- The package is fully documented
- Virtual environment files are provided
- Extensive testing of all modules with GH Actions and Codecov has been performed
- A code of conduct and contribution guidelines are included
