"""Output helper tools for future agent integrations.

The current pipeline writes final PRD, roadmap, and priority matrix files through
`src.output.prd_generator.PRDGenerator`. This module is intentionally lightweight
so the documented package path exists for extension work.
"""

from src.output.prd_generator import PRDGenerator

__all__ = ["PRDGenerator"]
