import logging
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from backend.config import settings

logger = logging.getLogger(__name__)

db_url = settings.DATABASE_URL
connect_args = {}

if db_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

try:
    engine = create_engine(db_url, connect_args=connect_args)
    # Test connection
    with engine.connect() as conn:
        pass
except Exception as e:
    logger.warning(f"Could not connect to configured DATABASE_URL ({db_url}): {e}. Falling back to SQLite.")
    db_url = "sqlite:///./cloudvision.db"
    connect_args = {"check_same_thread": False}
    engine = create_engine(db_url, connect_args=connect_args)

class Base(DeclarativeBase):
    pass

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)
    migrate_job_columns()
    # Seed default demo admin users if missing
    try:
        from backend.models import User
        from backend.auth import hash_password
        db = SessionLocal()
        admin_user = db.query(User).filter(User.email == "admin@cloudvision.ai").first()
        if not admin_user:
            admin_user = User(
                name="Demo Scientist",
                email="admin@cloudvision.ai",
                hashed_password=hash_password("password123")
            )
            db.add(admin_user)
        else:
            admin_user.hashed_password = hash_password("password123")

        analyst_user = db.query(User).filter(User.email == "analyst@liss4.org").first()
        if analyst_user:
            analyst_user.hashed_password = hash_password("password123")

        db.commit()
        db.close()
    except Exception as e:
        logger.warning(f"Could not seed default user: {e}")


def migrate_job_columns():
    """Add new job columns to existing SQLite and PostgreSQL databases."""
    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            for column_name in ("rmse", "sam", "ergas"):
                connection.execute(
                    text(f"ALTER TABLE jobs ADD COLUMN IF NOT EXISTS {column_name} FLOAT")
                )
        return

    if engine.dialect.name != "sqlite":
        return

    inspector = inspect(engine)
    existing = {column["name"] for column in inspector.get_columns("jobs")}
    additions = {
        "comparison_image": "TEXT",
        "psnr": "FLOAT",
        "ssim": "FLOAT",
        "rmse": "FLOAT",
        "sam": "FLOAT",
        "ergas": "FLOAT",
        "use_rag": "BOOLEAN DEFAULT 1",
        "rag_context": "TEXT",
        "rag_similarity_score": "FLOAT",
    }

    with engine.begin() as connection:
        for column_name, column_type in additions.items():
            if column_name not in existing:
                connection.execute(text(f"ALTER TABLE jobs ADD COLUMN {column_name} {column_type}"))
                logger.info("Added missing jobs.%s column", column_name)
