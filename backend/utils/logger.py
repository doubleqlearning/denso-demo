import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


class Logger:
    """
    A flexible logger helper class that provides easy-to-use logging functionality
    with support for both console and file logging.
    """
    
    def __init__(
        self,
        name: str = "voice-agent-project",
        level: Union[str, int] = logging.INFO,
        log_to_file: bool = True,
        log_dir: str = "logs",
        log_filename: Optional[str] = None,
        console_format: Optional[str] = None,
        file_format: Optional[str] = None,
    ):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: Whether to log to file
            log_dir: Directory to store log files
            log_filename: Custom log filename (defaults to timestamped filename)
            console_format: Custom console log format
            file_format: Custom file log format
        """
        self.name = name
        self.level = level
        self.log_to_file = log_to_file
        self.log_dir = log_dir
        self.log_filename = log_filename
        
        # Default formats
        self.console_format = console_format or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.file_format = file_format or "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        
        # Initialize logger
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up and configure the logger."""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        
        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_formatter = logging.Formatter(self.console_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler (if enabled)
        if self.log_to_file:
            file_handler = self._create_file_handler()
            if file_handler:
                logger.addHandler(file_handler)
        
        return logger
    
    def _create_file_handler(self) -> Optional[logging.FileHandler]:
        """Create and configure file handler."""
        try:
            # Create log directory if it doesn't exist
            log_path = Path(self.log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            
            # Generate filename if not provided
            if not self.log_filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.log_filename = f"{self.name}_{timestamp}.log"
            
            log_file_path = log_path / self.log_filename
            
            # Create file handler
            file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
            file_handler.setLevel(self.level)
            file_formatter = logging.Formatter(self.file_format)
            file_handler.setFormatter(file_formatter)
            
            return file_handler
            
        except Exception as e:
            print(f"Warning: Could not create file handler: {e}")
            return None
    
    def debug(self, message: str, *args, **kwargs):
        """Log debug message."""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log info message."""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log warning message."""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log error message."""
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log critical message."""
        self.logger.critical(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, *args, **kwargs)
    
    def set_level(self, level: Union[str, int]):
        """Change logging level."""
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)


# Global logger instance for easy access
_default_logger = None


def get_logger(
    name: str = "thesis-data-platform",
    level: Union[str, int] = logging.INFO,
    **kwargs
) -> Logger:
    """
    Get a logger instance. Creates a default logger if none exists.
    
    Args:
        name: Logger name
        level: Logging level
        **kwargs: Additional arguments for Logger initialization
    
    Returns:
        Logger instance
    """
    global _default_logger
    
    if _default_logger is None:
        _default_logger = Logger(name=name, level=level, **kwargs)
    
    return _default_logger


def setup_logger(
    name: str = "thesis-data-platform",
    level: Union[str, int] = logging.INFO,
    **kwargs
) -> Logger:
    """
    Set up and return a new logger instance.
    
    Args:
        name: Logger name
        level: Logging level
        **kwargs: Additional arguments for Logger initialization
    
    Returns:
        Logger instance
    """
    return Logger(name=name, level=level, **kwargs)


# Convenience functions for quick logging
def debug(message: str, *args, **kwargs):
    """Quick debug logging."""
    get_logger().debug(message, *args, **kwargs)


def info(message: str, *args, **kwargs):
    """Quick info logging."""
    get_logger().info(message, *args, **kwargs)


def warning(message: str, *args, **kwargs):
    """Quick warning logging."""
    get_logger().warning(message, *args, **kwargs)


def error(message: str, *args, **kwargs):
    """Quick error logging."""
    get_logger().error(message, *args, **kwargs)


def critical(message: str, *args, **kwargs):
    """Quick critical logging."""
    get_logger().critical(message, *args, **kwargs)


def exception(message: str, *args, **kwargs):
    """Quick exception logging."""
    get_logger().exception(message, *args, **kwargs) 