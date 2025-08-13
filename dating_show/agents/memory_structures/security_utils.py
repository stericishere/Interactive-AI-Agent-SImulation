"""
Security utilities for Enhanced PIANO Memory Architecture
Provides input validation, path sanitization, and security protections.
"""

import os
import re
import html
from pathlib import Path
from typing import Any, Optional


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


class SecurityValidator:
    """Security validation utilities for memory systems."""
    
    # Maximum allowed input lengths
    MAX_CONTENT_LENGTH = 10000
    MAX_FILEPATH_LENGTH = 255
    
    # Allowed file extensions for persistence
    ALLOWED_EXTENSIONS = {'.json', '.txt', '.log'}
    
    # Dangerous path patterns
    DANGEROUS_PATTERNS = [
        r'\.\./',           # Directory traversal
        r'\.\.\/',          # Unix-style traversal  
        r'\.\.\\',          # Windows-style traversal
        r'^[a-z]:[/\\]',    # Windows absolute paths
        r'^[/\\]',          # Unix absolute paths
        r'\\\\[^\\]+\\',    # UNC paths
        r'/dev/',           # Device files
        r'/proc/',          # Process files
        r'/sys/',           # System files
        r'system32',        # Windows system
        r'windows',         # Windows directory
        r'Program Files',   # Windows programs
    ]
    
    @classmethod
    def validate_content(cls, content: str, field_name: str = "content") -> str:
        """
        Validate and sanitize text content.
        
        Args:
            content: Content to validate
            field_name: Name of field for error messages
            
        Returns:
            Sanitized content
            
        Raises:
            SecurityError: If content is invalid or dangerous
        """
        if not isinstance(content, str):
            raise SecurityError(f"{field_name} must be a string, got {type(content)}")
        
        if len(content) > cls.MAX_CONTENT_LENGTH:
            raise SecurityError(f"{field_name} exceeds maximum length of {cls.MAX_CONTENT_LENGTH} chars")
        
        # Check for script tags and other XSS patterns
        if cls._contains_script_tags(content):
            raise SecurityError(f"{field_name} contains potentially dangerous script tags")
        
        # HTML escape the content to prevent XSS
        sanitized = html.escape(content)
        
        return sanitized
    
    @classmethod
    def validate_importance(cls, importance: Any) -> float:
        """
        Validate importance value.
        
        Args:
            importance: Importance value to validate
            
        Returns:
            Valid importance float
            
        Raises:
            SecurityError: If importance is invalid
        """
        if importance is None:
            raise SecurityError("Importance cannot be None")
        
        if isinstance(importance, str):
            try:
                importance = float(importance)
            except ValueError:
                raise SecurityError(f"Cannot convert importance '{importance}' to float")
        
        if not isinstance(importance, (int, float)):
            raise SecurityError(f"Importance must be numeric, got {type(importance)}")
        
        if not (0.0 <= importance <= 1.0):
            raise SecurityError(f"Importance must be between 0.0 and 1.0, got {importance}")
        
        return float(importance)
    
    @classmethod
    def validate_filepath(cls, filepath: str) -> str:
        """
        Validate and sanitize file path to prevent directory traversal.
        
        Args:
            filepath: File path to validate
            
        Returns:
            Safe file path
            
        Raises:
            SecurityError: If path is dangerous
        """
        if not isinstance(filepath, str):
            raise SecurityError(f"Filepath must be string, got {type(filepath)}")
        
        if len(filepath) > cls.MAX_FILEPATH_LENGTH:
            raise SecurityError(f"Filepath exceeds maximum length of {cls.MAX_FILEPATH_LENGTH}")
        
        # Check for dangerous patterns
        filepath_lower = filepath.lower()
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, filepath_lower, re.IGNORECASE):
                raise SecurityError(f"Filepath contains dangerous pattern: {pattern}")
        
        # Resolve path and check it's within allowed directory
        try:
            resolved_path = Path(filepath).resolve()
            
            # Ensure we're working within current directory or designated safe areas
            current_dir = Path.cwd()
            
            # Check if path is within current directory tree
            if not str(resolved_path).startswith(str(current_dir)):
                raise SecurityError(f"Filepath outside allowed directory: {resolved_path}")
            
        except (OSError, ValueError) as e:
            raise SecurityError(f"Invalid filepath: {e}")
        
        # Check file extension
        path_obj = Path(filepath)
        if path_obj.suffix.lower() not in cls.ALLOWED_EXTENSIONS:
            raise SecurityError(f"File extension not allowed: {path_obj.suffix}")
        
        return str(resolved_path)
    
    @classmethod
    def _contains_script_tags(cls, content: str) -> bool:
        """Check if content contains script tags or other XSS patterns."""
        dangerous_patterns = [
            r'<script[^>]*>',
            r'</script>',
            r'javascript:',
            r'on\w+\s*=',  # onclick, onload, etc.
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>',
        ]
        
        content_lower = content.lower()
        for pattern in dangerous_patterns:
            if re.search(pattern, content_lower, re.IGNORECASE):
                return True
        
        return False
    
    @classmethod
    def sanitize_memory_data(cls, content: str, memory_type: str, importance: Any, 
                           context: Optional[dict] = None) -> tuple:
        """
        Sanitize all memory data inputs.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            importance: Importance value
            context: Optional context data
            
        Returns:
            Tuple of (sanitized_content, validated_type, validated_importance, safe_context)
            
        Raises:
            SecurityError: If any input is invalid
        """
        # Validate and sanitize content
        safe_content = cls.validate_content(content, "content")
        
        # Validate memory type
        safe_type = cls.validate_content(memory_type, "memory_type")
        
        # Validate importance
        safe_importance = cls.validate_importance(importance)
        
        # Sanitize context if provided
        safe_context = None
        if context is not None:
            if not isinstance(context, dict):
                raise SecurityError(f"Context must be dict, got {type(context)}")
            
            safe_context = {}
            for key, value in context.items():
                if isinstance(value, str):
                    safe_context[key] = cls.validate_content(value, f"context.{key}")
                elif isinstance(value, (int, float, bool)):
                    safe_context[key] = value
                elif isinstance(value, list):
                    # Sanitize list values
                    safe_list = []
                    for item in value:
                        if isinstance(item, str):
                            safe_list.append(cls.validate_content(item, f"context.{key}[item]"))
                        else:
                            safe_list.append(item)
                    safe_context[key] = safe_list
                else:
                    # Skip dangerous types
                    continue
        
        return safe_content, safe_type, safe_importance, safe_context