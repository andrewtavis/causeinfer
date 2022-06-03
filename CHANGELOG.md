# Changelog

causeinfer tries to follow [semantic versioning](https://semver.org/), a MAJOR.MINOR.PATCH version where increments are made of the:

- MAJOR version when we make incompatible API changes
- MINOR version when we add functionality in a backwards compatible manner
- PATCH version when we make backwards compatible bug fixes

# causeinfer 1.0.1 (June 3rd, 2022)

- Updates source code files with direct references to codes they're based on.

# causeinfer 1.0.0 (December 28th, 2021)

- Release switches causeinfer over to [semantic versioning](https://semver.org/) and indicates that it is stable

# causeinfer 0.1.2 (April 4th, 2021)

Changes include:

- An src structure has been adopted to improve organization and testing
- Users are now able to implement the following models:
  - Reflective Uplift (Shaar 2016)
  - Pessimistic Uplift (Shaar 2016)
- The contribution guidelines have been expanded
- Code quality checks via Codacy have been added
- Extensive code formatting has been done to improve quality and style
- Bug fixes and a more explicit use of exceptions

# causeinfer 0.1.0 (Feb 25th, 2021)

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
