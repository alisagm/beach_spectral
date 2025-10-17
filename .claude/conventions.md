# Beach Spectral Classifier - Development Conventions

## Platform Requirements

### Windows Compatibility
This project is developed and deployed on **Windows** systems. All code must be Windows-compatible.

### Character Encoding

**CRITICAL:** All text output (especially logging) must use **ASCII-compatible encoding**.

#### Why ASCII?
- Windows console and file I/O has inconsistent Unicode support
- Logging to files can fail with Unicode characters on Windows
- ASCII ensures maximum compatibility across all Windows environments

#### Rules for Code:
1. **No Unicode characters in code or strings**
   - ❌ BAD: `class1 → class2` (Unicode arrow U+2192)
   - ✅ GOOD: `class1 -> class2` (ASCII hyphen + greater-than)

2. **Common Unicode to ASCII replacements:**
   - `→` (arrow) → `->`
   - `±` (plus-minus) → `+-`
   - `≈` (approximately) → `~`
   - `≠` (not equal) → `!=`
   - `≤` (less than or equal) → `<=`
   - `≥` (greater than or equal) → `>=`

3. **File I/O encoding:**
   - Logging: Use `encoding='ascii', errors='replace'`
   - CSV files: Use default encoding (UTF-8 is OK for data files)
   - Log files: **Must be ASCII-compatible** (see utils.py:50)

#### Example - Correct Logging Setup:
```python
import logging

# Correct way to set up file logging on Windows
file_handler = logging.FileHandler(log_file, encoding='ascii', errors='replace')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
```

#### Example - Correct String Formatting:
```python
# ❌ BAD - Unicode arrow
logger.info(f"Transition: {from_class}→{to_class}")

# ✅ GOOD - ASCII arrow
logger.info(f"Transition: {from_class}->{to_class}")
```

## Code Style

### Python Version
- Python 3.9+ required
- Type hints encouraged for public functions

### Logging
- Use structured logging with appropriate levels:
  - DEBUG: Detailed diagnostic information
  - INFO: General progress and milestones
  - WARNING: Potential issues or unexpected behavior
  - ERROR: Errors that prevent operation
- File logging: INFO and above (or DEBUG if verbose=True)
- Console logging: WARNING and above (reduce noise)

### Documentation
- Docstrings required for all public functions/classes
- Use NumPy/Google style docstrings
- Include Args, Returns, and Raises sections

### Naming Conventions
- Functions: `lowercase_with_underscores`
- Classes: `PascalCase`
- Constants: `UPPER_CASE_WITH_UNDERSCORES`
- Private methods: `_leading_underscore`

## Project Structure

### Core Modules
- `spectral_classifier/` - Main classification pipeline
  - `config.py` - Configuration and thresholds
  - `data_io.py` - Raster and transect loading
  - `sampler.py` - Spectral sampling along transects
  - `features.py` - Feature extraction
  - `classifier.py` - Landcover classification
  - `transition.py` - Boundary detection
  - `visualization.py` - Plotting and visualization
  - `utils.py` - Utility functions (logging, export, etc.)

### Supporting Directories
- `tools/` - Data generation scripts
- `validation/` - Validation and testing
- `analysis/` - Analysis scripts and outputs
- `tests/` - Unit and integration tests

## Git Workflow

### Commit Messages
- Use clear, descriptive commit messages
- Reference issue numbers when applicable
- Follow conventional commit format when possible:
  - `fix: correct Unicode encoding in logging`
  - `feat: add fallback detection mode`
  - `docs: update conventions.md`

### Branch Strategy
- `main` - Production-ready code
- Feature branches - For development work

## Testing

### Manual Testing
- Always test on Windows before committing
- Verify log files are created without encoding errors
- Check console output for garbled characters

### Validation
- Run validation suite after major changes
- Compare against manual classifications
- Check precision/recall metrics

## Common Pitfalls

### Unicode Encoding Errors
**Symptom:** `UnicodeEncodeError: 'charmap' codec can't encode character`

**Solution:**
1. Check for Unicode characters in logging statements
2. Replace with ASCII equivalents (see table above)
3. Ensure file handlers use `encoding='ascii', errors='replace'`

### Path Handling
- Use `pathlib.Path` for all path operations
- Always use forward slashes or Path objects (not backslashes in strings)
- Test with spaces in paths

## Questions?

For questions about conventions or to suggest improvements, create an issue or discuss with the team.

---

Last updated: 2025-01-17
