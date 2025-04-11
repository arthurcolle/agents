#!/usr/bin/env python3
"""
Filesystem Tools - Safe file system operations for AI agents

Provides tools for an AI agent to work with the file system:
- File reading and writing with safety controls
- Directory operations (list, create, search)
- File type detection and handling
- Path manipulation and validation

All operations include safety checks to prevent unintended 
modifications or access to sensitive areas.
"""

import os
import sys
import re
import glob
import shutil
import json
import hashlib
import mimetypes
import pathlib
import stat
import tempfile
import time
from typing import Dict, List, Any, Optional, Tuple, Set, Union

# === SAFETY CONTROLS ===
class FilesystemSafety:
    """
    Manages safety controls for filesystem operations
    """
    def __init__(self, 
                root_dir: Optional[str] = None,
                protected_paths: Optional[List[str]] = None,
                max_file_size: int = 50 * 1024 * 1024):  # 50MB default
        """
        Initialize safety controls
        
        Args:
            root_dir: Root directory to restrict operations to (None for no restriction)
            protected_paths: List of paths that cannot be modified
            max_file_size: Maximum file size in bytes for read/write operations
        """
        self.root_dir = os.path.abspath(root_dir) if root_dir else None
        self.protected_paths = set()
        
        # Add default protected paths
        default_protected = [
            "/etc", "/var", "/bin", "/sbin", "/usr/bin", "/usr/sbin",
            "/System", "/Users/*/Library", "/Library", 
            "~/.ssh", "~/.aws", "~/.config",
            "*/passwords*", "*/credential*", "*/secret*", "*/key*"
        ]
        
        if protected_paths:
            default_protected.extend(protected_paths)
            
        # Expand and normalize paths
        for path in default_protected:
            # Handle glob patterns
            if any(c in path for c in ["*", "?", "["]):
                matching = glob.glob(os.path.expanduser(path))
                self.protected_paths.update(os.path.abspath(p) for p in matching)
            else:
                # Regular path
                self.protected_paths.add(os.path.abspath(os.path.expanduser(path)))
                
        self.max_file_size = max_file_size
        
    def is_path_allowed(self, path: str, write_access: bool = False) -> Tuple[bool, str]:
        """
        Check if a path is allowed for access
        
        Args:
            path: Path to check
            write_access: Whether write access is requested
            
        Returns:
            Tuple of (allowed, reason)
        """
        # Normalize the path
        real_path = os.path.abspath(os.path.expanduser(path))
        
        # Check if within root directory (if configured)
        if self.root_dir and not real_path.startswith(self.root_dir):
            return False, f"Path outside of allowed root directory: {self.root_dir}"
            
        # Check against protected paths
        for protected in self.protected_paths:
            if real_path.startswith(protected) or real_path == protected:
                return False, f"Access to protected path not allowed: {protected}"
                
        # Additional checks for write access
        if write_access:
            # Don't allow writing to system directories or binaries
            if any(real_path.startswith(p) for p in [
                "/bin", "/sbin", "/usr/bin", "/usr/sbin", "/etc", 
                "/System", "/Library"
            ]):
                return False, "Write access to system directories not allowed"
                
            # Check for suspicious filename patterns
            suspicious_patterns = [
                r"passwd", r"shadow", r"\.ssh", r"authorized_keys",
                r"id_rsa", r"\.aws", r"credentials", r"\.env", 
                r"config\.sys", r"(secret|token|key|password)\."
            ]
            
            basename = os.path.basename(real_path)
            for pattern in suspicious_patterns:
                if re.search(pattern, basename, re.IGNORECASE):
                    return False, f"Writing to potentially sensitive file not allowed: {basename}"
                    
        return True, ""
        
    def check_file_size(self, path: str) -> Tuple[bool, str]:
        """
        Check if a file is within the size limit
        
        Args:
            path: Path to check
            
        Returns:
            Tuple of (allowed, reason)
        """
        if not os.path.exists(path):
            return True, ""
            
        size = os.path.getsize(path)
        if size > self.max_file_size:
            return False, f"File exceeds maximum size limit ({size} > {self.max_file_size} bytes)"
            
        return True, ""

# === FILE OPERATIONS ===
class FileOperations:
    """
    Operations for reading and writing files
    """
    def __init__(self, safety: FilesystemSafety):
        """
        Initialize with safety controls
        
        Args:
            safety: FilesystemSafety instance
        """
        self.safety = safety
        
    def read_file(self, 
                path: str, 
                binary: bool = False, 
                offset: int = 0, 
                limit: Optional[int] = None) -> Dict:
        """
        Read a file safely
        
        Args:
            path: Path to file
            binary: Whether to read in binary mode
            offset: Byte/character offset to start from
            limit: Maximum bytes/characters to read
            
        Returns:
            Dictionary with file content or error info
        """
        # Normalize the path
        path = os.path.expanduser(path)
        
        # Check if path is allowed
        allowed, reason = self.safety.is_path_allowed(path)
        if not allowed:
            return {
                "status": "error",
                "message": reason
            }
            
        # Check if file exists
        if not os.path.exists(path):
            return {
                "status": "error",
                "message": f"File not found: {path}"
            }
            
        # Check if it's a directory
        if os.path.isdir(path):
            return {
                "status": "error",
                "message": f"Path is a directory, not a file: {path}"
            }
            
        # Check file size
        allowed, reason = self.safety.check_file_size(path)
        if not allowed:
            return {
                "status": "error",
                "message": reason
            }
            
        try:
            # Detect file type
            mime_type, encoding = mimetypes.guess_type(path)
            
            # Read the file
            mode = 'rb' if binary else 'r'
            with open(path, mode) as f:
                if offset > 0:
                    f.seek(offset)
                    
                if limit is not None:
                    content = f.read(limit)
                else:
                    content = f.read()
                    
            # Get file stats
            stats = os.stat(path)
            
            return {
                "status": "success",
                "content": content,
                "size": stats.st_size,
                "path": path,
                "binary": binary,
                "mime_type": mime_type,
                "modified": stats.st_mtime,
                "created": stats.st_ctime
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error reading file: {str(e)}"
            }
            
    def write_file(self, 
                 path: str, 
                 content: Union[str, bytes], 
                 append: bool = False,
                 backup: bool = True) -> Dict:
        """
        Write to a file safely
        
        Args:
            path: Path to file
            content: Content to write
            append: Whether to append to the file
            backup: Whether to create a backup
            
        Returns:
            Dictionary with status info
        """
        # Normalize the path
        path = os.path.expanduser(path)
        
        # Check if path is allowed for writing
        allowed, reason = self.safety.is_path_allowed(path, write_access=True)
        if not allowed:
            return {
                "status": "error",
                "message": reason
            }
            
        # Determine if content is binary
        is_binary = isinstance(content, bytes)
        
        try:
            # Create directories if needed
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                
            # Create backup if requested and file exists
            if backup and os.path.exists(path):
                backup_path = f"{path}.bak"
                shutil.copy2(path, backup_path)
                
            # Write the file
            mode = 'ab' if append and is_binary else 'wb' if is_binary else 'a' if append else 'w'
            with open(path, mode) as f:
                f.write(content)
                
            # Get file stats
            stats = os.stat(path)
            
            return {
                "status": "success",
                "path": path,
                "size": stats.st_size,
                "modified": stats.st_mtime,
                "backup": backup_path if backup and os.path.exists(path) else None
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error writing file: {str(e)}"
            }
            
    def copy_file(self, 
                source: str, 
                destination: str, 
                overwrite: bool = False) -> Dict:
        """
        Copy a file safely
        
        Args:
            source: Source file path
            destination: Destination file path
            overwrite: Whether to overwrite existing files
            
        Returns:
            Dictionary with status info
        """
        # Normalize the paths
        source = os.path.expanduser(source)
        destination = os.path.expanduser(destination)
        
        # Check if source path is allowed
        allowed, reason = self.safety.is_path_allowed(source)
        if not allowed:
            return {
                "status": "error",
                "message": f"Source: {reason}"
            }
            
        # Check if destination path is allowed for writing
        allowed, reason = self.safety.is_path_allowed(destination, write_access=True)
        if not allowed:
            return {
                "status": "error",
                "message": f"Destination: {reason}"
            }
            
        # Check if source exists
        if not os.path.exists(source):
            return {
                "status": "error",
                "message": f"Source file not found: {source}"
            }
            
        # Check if destination already exists
        if os.path.exists(destination) and not overwrite:
            return {
                "status": "error",
                "message": f"Destination already exists: {destination}"
            }
            
        try:
            # Create destination directory if needed
            dest_dir = os.path.dirname(destination)
            if dest_dir and not os.path.exists(dest_dir):
                os.makedirs(dest_dir, exist_ok=True)
                
            # Copy the file
            shutil.copy2(source, destination)
            
            # Get file stats
            stats = os.stat(destination)
            
            return {
                "status": "success",
                "source": source,
                "destination": destination,
                "size": stats.st_size,
                "modified": stats.st_mtime
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error copying file: {str(e)}"
            }
            
    def delete_file(self, path: str, secure: bool = False) -> Dict:
        """
        Delete a file safely
        
        Args:
            path: Path to file
            secure: Whether to securely overwrite before deleting
            
        Returns:
            Dictionary with status info
        """
        # Normalize the path
        path = os.path.expanduser(path)
        
        # Check if path is allowed for writing
        allowed, reason = self.safety.is_path_allowed(path, write_access=True)
        if not allowed:
            return {
                "status": "error",
                "message": reason
            }
            
        # Check if file exists
        if not os.path.exists(path):
            return {
                "status": "error",
                "message": f"File not found: {path}"
            }
            
        # Check if it's a directory
        if os.path.isdir(path):
            return {
                "status": "error",
                "message": f"Path is a directory, not a file: {path}"
            }
            
        try:
            # Create a backup
            backup_path = f"{path}.deleted.bak"
            shutil.copy2(path, backup_path)
            
            # Secure deletion if requested
            if secure and os.path.exists(path):
                # Get file size
                file_size = os.path.getsize(path)
                
                # Open file and overwrite with random data
                with open(path, 'wb') as f:
                    # First pass: zeros
                    f.write(b'\x00' * file_size)
                    f.flush()
                    os.fsync(f.fileno())
                    
                    # Second pass: ones
                    f.seek(0)
                    f.write(b'\xFF' * file_size)
                    f.flush()
                    os.fsync(f.fileno())
                    
                    # Third pass: random
                    f.seek(0)
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())
                    
            # Delete the file
            os.remove(path)
            
            return {
                "status": "success",
                "path": path,
                "backup": backup_path,
                "secure_delete": secure
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error deleting file: {str(e)}"
            }

# === DIRECTORY OPERATIONS ===
class DirectoryOperations:
    """
    Operations for working with directories
    """
    def __init__(self, safety: FilesystemSafety):
        """
        Initialize with safety controls
        
        Args:
            safety: FilesystemSafety instance
        """
        self.safety = safety
        
    def list_directory(self, 
                      path: str, 
                      recursive: bool = False,
                      include_hidden: bool = False,
                      include_details: bool = False) -> Dict:
        """
        List contents of a directory
        
        Args:
            path: Directory path
            recursive: Whether to list subdirectories
            include_hidden: Whether to include hidden files/dirs
            include_details: Whether to include detailed stats
            
        Returns:
            Dictionary with directory contents
        """
        # Normalize the path
        path = os.path.expanduser(path)
        
        # Check if path is allowed
        allowed, reason = self.safety.is_path_allowed(path)
        if not allowed:
            return {
                "status": "error",
                "message": reason
            }
            
        # Check if directory exists
        if not os.path.exists(path):
            return {
                "status": "error",
                "message": f"Directory not found: {path}"
            }
            
        # Check if it's a directory
        if not os.path.isdir(path):
            return {
                "status": "error",
                "message": f"Path is a file, not a directory: {path}"
            }
            
        try:
            results = []
            
            if recursive:
                # Walk the directory tree
                for root, dirs, files in os.walk(path):
                    # Skip hidden directories if not included
                    if not include_hidden:
                        dirs[:] = [d for d in dirs if not d.startswith('.')]
                        
                    # Process files
                    for file in files:
                        # Skip hidden files if not included
                        if not include_hidden and file.startswith('.'):
                            continue
                            
                        file_path = os.path.join(root, file)
                        
                        # Get detailed info if requested
                        if include_details:
                            stats = os.stat(file_path)
                            mime_type, encoding = mimetypes.guess_type(file_path)
                            
                            results.append({
                                "name": file,
                                "path": file_path,
                                "type": "file",
                                "size": stats.st_size,
                                "modified": stats.st_mtime,
                                "created": stats.st_ctime,
                                "mime_type": mime_type
                            })
                        else:
                            results.append({
                                "name": file,
                                "path": file_path,
                                "type": "file"
                            })
                            
                    # Process directories (except the root)
                    if root != path:
                        rel_path = os.path.relpath(root, path)
                        
                        # Skip if this is a hidden directory
                        if not include_hidden and any(part.startswith('.') for part in rel_path.split(os.sep)):
                            continue
                            
                        # Get detailed info if requested
                        if include_details:
                            stats = os.stat(root)
                            
                            results.append({
                                "name": os.path.basename(root),
                                "path": root,
                                "type": "directory",
                                "size": stats.st_size,
                                "modified": stats.st_mtime,
                                "created": stats.st_ctime
                            })
                        else:
                            results.append({
                                "name": os.path.basename(root),
                                "path": root,
                                "type": "directory"
                            })
            else:
                # List only the current directory
                for item in os.listdir(path):
                    # Skip hidden items if not included
                    if not include_hidden and item.startswith('.'):
                        continue
                        
                    item_path = os.path.join(path, item)
                    item_type = "directory" if os.path.isdir(item_path) else "file"
                    
                    # Get detailed info if requested
                    if include_details:
                        stats = os.stat(item_path)
                        mime_type = None
                        
                        if item_type == "file":
                            mime_type, encoding = mimetypes.guess_type(item_path)
                            
                        results.append({
                            "name": item,
                            "path": item_path,
                            "type": item_type,
                            "size": stats.st_size,
                            "modified": stats.st_mtime,
                            "created": stats.st_ctime,
                            "mime_type": mime_type if item_type == "file" else None
                        })
                    else:
                        results.append({
                            "name": item,
                            "path": item_path,
                            "type": item_type
                        })
                        
            return {
                "status": "success",
                "path": path,
                "items": results,
                "count": len(results)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error listing directory: {str(e)}"
            }
            
    def create_directory(self, path: str, parents: bool = True) -> Dict:
        """
        Create a directory safely
        
        Args:
            path: Directory path
            parents: Whether to create parent directories
            
        Returns:
            Dictionary with status info
        """
        # Normalize the path
        path = os.path.expanduser(path)
        
        # Check if path is allowed for writing
        allowed, reason = self.safety.is_path_allowed(path, write_access=True)
        if not allowed:
            return {
                "status": "error",
                "message": reason
            }
            
        # Check if directory already exists
        if os.path.exists(path):
            if os.path.isdir(path):
                return {
                    "status": "error",
                    "message": f"Directory already exists: {path}"
                }
            else:
                return {
                    "status": "error",
                    "message": f"Path exists but is a file, not a directory: {path}"
                }
                
        try:
            # Create the directory
            if parents:
                os.makedirs(path)
            else:
                parent_dir = os.path.dirname(path)
                if not os.path.exists(parent_dir):
                    return {
                        "status": "error",
                        "message": f"Parent directory does not exist: {parent_dir}"
                    }
                os.mkdir(path)
                
            # Get directory stats
            stats = os.stat(path)
            
            return {
                "status": "success",
                "path": path,
                "created": stats.st_ctime
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error creating directory: {str(e)}"
            }
            
    def delete_directory(self, path: str, recursive: bool = False) -> Dict:
        """
        Delete a directory safely
        
        Args:
            path: Directory path
            recursive: Whether to delete contents recursively
            
        Returns:
            Dictionary with status info
        """
        # Normalize the path
        path = os.path.expanduser(path)
        
        # Check if path is allowed for writing
        allowed, reason = self.safety.is_path_allowed(path, write_access=True)
        if not allowed:
            return {
                "status": "error",
                "message": reason
            }
            
        # Check if directory exists
        if not os.path.exists(path):
            return {
                "status": "error",
                "message": f"Directory not found: {path}"
            }
            
        # Check if it's a directory
        if not os.path.isdir(path):
            return {
                "status": "error",
                "message": f"Path is a file, not a directory: {path}"
            }
            
        try:
            # Check if directory is empty
            if not recursive and os.listdir(path):
                return {
                    "status": "error",
                    "message": f"Directory not empty: {path}"
                }
                
            # Create a temporary backup directory
            backup_dir = tempfile.mkdtemp(prefix=f"{os.path.basename(path)}_backup_")
            
            # Copy contents to backup
            if recursive:
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        shutil.copytree(item_path, os.path.join(backup_dir, item))
                    else:
                        shutil.copy2(item_path, os.path.join(backup_dir, item))
                        
            # Delete the directory
            if recursive:
                shutil.rmtree(path)
            else:
                os.rmdir(path)
                
            return {
                "status": "success",
                "path": path,
                "backup": backup_dir,
                "recursive": recursive
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error deleting directory: {str(e)}"
            }
            
    def search_files(self, 
                    path: str, 
                    pattern: str, 
                    recursive: bool = True,
                    file_type: Optional[str] = None) -> Dict:
        """
        Search for files matching a pattern
        
        Args:
            path: Directory path to search in
            pattern: Glob pattern to match
            recursive: Whether to search subdirectories
            file_type: Filter by file type (extension)
            
        Returns:
            Dictionary with matching files
        """
        # Normalize the path
        path = os.path.expanduser(path)
        
        # Check if path is allowed
        allowed, reason = self.safety.is_path_allowed(path)
        if not allowed:
            return {
                "status": "error",
                "message": reason
            }
            
        # Check if directory exists
        if not os.path.exists(path):
            return {
                "status": "error",
                "message": f"Directory not found: {path}"
            }
            
        # Check if it's a directory
        if not os.path.isdir(path):
            return {
                "status": "error",
                "message": f"Path is a file, not a directory: {path}"
            }
            
        try:
            # Build the glob pattern
            if recursive:
                glob_pattern = os.path.join(path, '**', pattern)
            else:
                glob_pattern = os.path.join(path, pattern)
                
            # Find matching files
            if recursive:
                matches = glob.glob(glob_pattern, recursive=True)
            else:
                matches = glob.glob(glob_pattern)
                
            # Filter by file type if specified
            if file_type:
                if not file_type.startswith('.'):
                    file_type = f".{file_type}"
                matches = [m for m in matches if m.endswith(file_type)]
                
            # Filter out directories
            matches = [m for m in matches if os.path.isfile(m)]
            
            results = []
            for match in matches:
                stats = os.stat(match)
                mime_type, encoding = mimetypes.guess_type(match)
                
                results.append({
                    "name": os.path.basename(match),
                    "path": match,
                    "size": stats.st_size,
                    "modified": stats.st_mtime,
                    "mime_type": mime_type
                })
                
            return {
                "status": "success",
                "pattern": pattern,
                "path": path,
                "matches": results,
                "count": len(results)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error searching files: {str(e)}"
            }

# === FILE INFO ===
class FileInfo:
    """
    Get information about files
    """
    def __init__(self, safety: FilesystemSafety):
        """
        Initialize with safety controls
        
        Args:
            safety: FilesystemSafety instance
        """
        self.safety = safety
        
    def get_file_info(self, path: str) -> Dict:
        """
        Get detailed information about a file
        
        Args:
            path: Path to file
            
        Returns:
            Dictionary with file information
        """
        # Normalize the path
        path = os.path.expanduser(path)
        
        # Check if path is allowed
        allowed, reason = self.safety.is_path_allowed(path)
        if not allowed:
            return {
                "status": "error",
                "message": reason
            }
            
        # Check if file exists
        if not os.path.exists(path):
            return {
                "status": "error",
                "message": f"File not found: {path}"
            }
            
        try:
            # Get file stats
            stats = os.stat(path)
            
            # Determine type
            is_dir = os.path.isdir(path)
            is_symlink = os.path.islink(path)
            is_file = os.path.isfile(path)
            
            # Get mime type
            mime_type = None
            if is_file:
                mime_type, encoding = mimetypes.guess_type(path)
                
            # Calculate permissions
            perms = stats.st_mode
            permission_str = stat.filemode(perms)
            
            # Calculate hash if it's a file
            file_hash = None
            if is_file and stats.st_size < self.safety.max_file_size:
                with open(path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                    
            # Return the info
            result = {
                "status": "success",
                "path": path,
                "name": os.path.basename(path),
                "type": "directory" if is_dir else "symlink" if is_symlink else "file",
                "size": stats.st_size,
                "permissions": permission_str,
                "mode": stats.st_mode,
                "owner": stats.st_uid,
                "group": stats.st_gid,
                "created": stats.st_ctime,
                "modified": stats.st_mtime,
                "accessed": stats.st_atime
            }
            
            if is_file:
                result["mime_type"] = mime_type
                result["hash_md5"] = file_hash
                
            if is_symlink:
                result["target"] = os.path.realpath(path)
                
            return result
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error getting file info: {str(e)}"
            }
            
    def get_path_info(self, path: str) -> Dict:
        """
        Get information about a path
        
        Args:
            path: Path to analyze
            
        Returns:
            Dictionary with path information
        """
        # Normalize the path
        path = os.path.expanduser(path)
        
        try:
            path_obj = pathlib.Path(path)
            
            return {
                "status": "success",
                "path": path,
                "exists": path_obj.exists(),
                "basename": path_obj.name,
                "dirname": str(path_obj.parent),
                "absolute": str(path_obj.absolute()),
                "extension": path_obj.suffix,
                "stem": path_obj.stem,
                "is_absolute": path_obj.is_absolute(),
                "parts": list(path_obj.parts)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error analyzing path: {str(e)}"
            }
            
    def check_path_access(self, path: str, mode: str = "r") -> Dict:
        """
        Check if a path is accessible
        
        Args:
            path: Path to check
            mode: Access mode (r=read, w=write, x=execute)
            
        Returns:
            Dictionary with access information
        """
        # Normalize the path
        path = os.path.expanduser(path)
        
        # Check if path is allowed by safety controls
        write_access = 'w' in mode
        allowed, reason = self.safety.is_path_allowed(path, write_access=write_access)
        
        if not allowed:
            return {
                "status": "error",
                "message": reason,
                "allowed": False
            }
            
        try:
            # Convert mode to access constants
            access_mode = 0
            if 'r' in mode:
                access_mode |= os.R_OK
            if 'w' in mode:
                access_mode |= os.W_OK
            if 'x' in mode:
                access_mode |= os.X_OK
                
            # Check access
            has_access = os.access(path, access_mode)
            
            if has_access:
                return {
                    "status": "success",
                    "path": path,
                    "mode": mode,
                    "access": True
                }
            else:
                return {
                    "status": "error",
                    "path": path,
                    "mode": mode,
                    "access": False,
                    "message": f"Access denied for mode {mode}"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error checking access: {str(e)}"
            }

# === UTILITY FUNCTIONS ===
def read_file(path: str, binary: bool = False, offset: int = 0, 
            limit: Optional[int] = None, root_dir: Optional[str] = None) -> Dict:
    """
    Read a file safely
    
    Args:
        path: Path to file
        binary: Whether to read in binary mode
        offset: Byte/character offset to start from
        limit: Maximum bytes/characters to read
        root_dir: Root directory to restrict operations to
        
    Returns:
        Dictionary with file content or error info
    """
    safety = FilesystemSafety(root_dir=root_dir)
    file_ops = FileOperations(safety)
    return file_ops.read_file(path, binary, offset, limit)

def write_file(path: str, content: Union[str, bytes], append: bool = False,
              backup: bool = True, root_dir: Optional[str] = None) -> Dict:
    """
    Write to a file safely
    
    Args:
        path: Path to file
        content: Content to write
        append: Whether to append to the file
        backup: Whether to create a backup
        root_dir: Root directory to restrict operations to
        
    Returns:
        Dictionary with status info
    """
    safety = FilesystemSafety(root_dir=root_dir)
    file_ops = FileOperations(safety)
    return file_ops.write_file(path, content, append, backup)

def list_directory(path: str, recursive: bool = False, include_hidden: bool = False,
                 include_details: bool = False, root_dir: Optional[str] = None) -> Dict:
    """
    List contents of a directory
    
    Args:
        path: Directory path
        recursive: Whether to list subdirectories
        include_hidden: Whether to include hidden files/dirs
        include_details: Whether to include detailed stats
        root_dir: Root directory to restrict operations to
        
    Returns:
        Dictionary with directory contents
    """
    safety = FilesystemSafety(root_dir=root_dir)
    dir_ops = DirectoryOperations(safety)
    return dir_ops.list_directory(path, recursive, include_hidden, include_details)

def search_files(path: str, pattern: str, recursive: bool = True,
               file_type: Optional[str] = None, root_dir: Optional[str] = None) -> Dict:
    """
    Search for files matching a pattern
    
    Args:
        path: Directory path to search in
        pattern: Glob pattern to match
        recursive: Whether to search subdirectories
        file_type: Filter by file type (extension)
        root_dir: Root directory to restrict operations to
        
    Returns:
        Dictionary with matching files
    """
    safety = FilesystemSafety(root_dir=root_dir)
    dir_ops = DirectoryOperations(safety)
    return dir_ops.search_files(path, pattern, recursive, file_type)

def get_file_info(path: str, root_dir: Optional[str] = None) -> Dict:
    """
    Get detailed information about a file
    
    Args:
        path: Path to file
        root_dir: Root directory to restrict operations to
        
    Returns:
        Dictionary with file information
    """
    safety = FilesystemSafety(root_dir=root_dir)
    file_info = FileInfo(safety)
    return file_info.get_file_info(path)

def is_path_allowed(path: str, write_access: bool = False, 
                  root_dir: Optional[str] = None) -> Dict:
    """
    Check if a path is allowed by safety controls
    
    Args:
        path: Path to check
        write_access: Whether write access is requested
        root_dir: Root directory to restrict operations to
        
    Returns:
        Dictionary with access information
    """
    safety = FilesystemSafety(root_dir=root_dir)
    allowed, reason = safety.is_path_allowed(path, write_access)
    
    return {
        "status": "success" if allowed else "error",
        "path": path,
        "allowed": allowed,
        "message": reason if not allowed else "Path is allowed",
        "write_access": write_access
    }

# === COMMAND-LINE INTERFACE ===
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Filesystem Tools")
    parser.add_argument("--read", help="Read a file")
    parser.add_argument("--write", help="Write to a file")
    parser.add_argument("--content", help="Content to write")
    parser.add_argument("--list", help="List a directory")
    parser.add_argument("--search", help="Search for files")
    parser.add_argument("--pattern", help="Pattern for search")
    parser.add_argument("--info", help="Get file info")
    parser.add_argument("--root-dir", help="Root directory to restrict operations to")
    parser.add_argument("--recursive", action="store_true", help="Recursive operation")
    parser.add_argument("--binary", action="store_true", help="Binary mode")
    parser.add_argument("--append", action="store_true", help="Append to file")
    parser.add_argument("--hidden", action="store_true", help="Include hidden files")
    parser.add_argument("--details", action="store_true", help="Include detailed info")
    
    args = parser.parse_args()
    
    if args.read:
        result = read_file(args.read, args.binary, root_dir=args.root_dir)
        print(json.dumps(result, indent=2))
    elif args.write and args.content:
        result = write_file(args.write, args.content, args.append, root_dir=args.root_dir)
        print(json.dumps(result, indent=2))
    elif args.list:
        result = list_directory(args.list, args.recursive, args.hidden, args.details, args.root_dir)
        print(json.dumps(result, indent=2))
    elif args.search and args.pattern:
        result = search_files(args.search, args.pattern, args.recursive, root_dir=args.root_dir)
        print(json.dumps(result, indent=2))
    elif args.info:
        result = get_file_info(args.info, args.root_dir)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()