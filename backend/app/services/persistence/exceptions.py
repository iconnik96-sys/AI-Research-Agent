class DatabaseError(Exception):
    """Base exception for all database persistence errors."""
    pass


class DatabaseConfigError(DatabaseError):
    """Raised when database configuration (e.g. DATABASE_URL) is missing or invalid."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when unable to connect or communicate with the database."""
    pass


class SessionNotFoundError(DatabaseError):
    """Raised when a research session is not found in the database."""
    pass
