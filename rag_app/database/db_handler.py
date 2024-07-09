from sqlmodel import SQLModel, create_engine, Session, select
from rag_app.database.schema import Sources
from rag_app.utils.logger import get_console_logger
import os
from dotenv import load_dotenv
import uuid
from datetime import datetime


class DataBaseHandler(): 
    """
    A class for managing the database.

    Attributes:
        sqlite_file_name (str): The SQLite file name for the database.
        logger (Logger): The logger for logging database operations.
        engine (Engine): The SQLAlchemy engine for the database.

    Methods:
        create_all_tables: Create all tables in the database.
        read_one: Read a single entry from the database by its hash_id.
        add_one: Add a single entry to the database.
        update_one: Update a single entry in the database by its hash_id.
        delete_one: Delete a single entry from the database by its id.
        add_many: Add multiple entries to the database.
        delete_many: Delete multiple entries from the database by their ids.
        read_all: Read all entries from the database, optionally filtered by a query.
        delete_all: Delete all entries from the database.
    """
    
    def __init__(
        self,
        sqlite_file_name = os.getenv('SOURCES_CACHE'),
        logger = get_console_logger("db_handler"),
        # *args,
        # **kwargs,
        ):
        self.sqlite_file_name = sqlite_file_name
        self.logger = logger
        
        sqlite_url = f"sqlite:///{self.sqlite_file_name}"
        self.engine = create_engine(sqlite_url, echo=False)
        
        
        self.session_id = str(uuid.uuid4())
        self.session_date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
    def create_all_tables(self) -> None:
        SQLModel.metadata.create_all(self.engine)
    
    def create_new_session(self) -> None:
        """creates a new session_id and date time
        
        """
        self.session_id = str(uuid.uuid4())
        self.session_date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


    def read_one(self,hash_id: dict):
        """
        Read a single entry from the database by its hash_id.

        Args:
            hash_id (dict): Dictionary containing the hash_id to search for.

        Returns:
            Sources: The matching entry from the database, or None if no match is found.
        """
        with Session(self.engine) as session:
            statement = select(Sources).where(Sources.hash_id == hash_id)
            sources = session.exec(statement).first()
            return sources


    def add_one(self,data: dict):
        """
        Add a single entry to the database.

        Args:
            data (dict): Dictionary containing the data for the new entry.

        Returns:
            Sources: The added entry, or None if the entry already exists.
        """
        with Session(self.engine) as session:
            if session.exec(
                select(Sources).where(Sources.hash_id == data.get("hash_id"))
            ).first():
                self.logger.warning(f"Item with hash_id {data.get('hash_id')} already exists")
                return None  # or raise an exception, or handle as needed
            sources = Sources(**data)
            session.add(sources)
            session.commit()
            session.refresh(sources)
            self.logger.info(f"Item with hash_id {data.get('hash_id')} added to the database")
            return sources


    def update_one(self,hash_id: dict, data: dict):
        """
        Update a single entry in the database by its hash_id.

        Args:
            hash_id (dict): Dictionary containing the hash_id to search for.
            data (dict): Dictionary containing the updated data for the entry.

        Returns:
            Sources: The updated entry, or None if no match is found.
        """
        with Session(self.engine) as session:
            # Check if the item with the given hash_id exists
            sources = session.exec(
                select(Sources).where(Sources.hash_id == hash_id)
            ).first()
            if not sources:
                self.logger.warning(f"No item with hash_id {hash_id} found for update")
                return None  # or raise an exception, or handle as needed
            for key, value in data.items():
                setattr(sources, key, value)
            session.commit()
            self.logger.info(f"Item with hash_id {hash_id} updated in the database")
            return sources


    def delete_one(self,id: int):
        """
        Delete a single entry from the database by its id.

        Args:
            id (int): The id of the entry to delete.

        Returns:
            None
        """
        with Session(self.engine) as session:
            # Check if the item with the given hash_id exists
            sources = session.exec(
                select(Sources).where(Sources.hash_id == id)
            ).first()
            if not sources:
                self.logger.warning(f"No item with hash_id {id} found for deletion")
                return None  # or raise an exception, or handle as needed
            session.delete(sources)
            session.commit()
            self.logger.info(f"Item with hash_id {id} deleted from the database")


    def add_many(self,data: list):
        """
        Add multiple entries to the database.

        Args:
            data (list): List of dictionaries, each containing the data for a new entry.

        Returns:
            None
        """
        with Session(self.engine) as session:
            for info in data:
                # Reuse add_one function for each item
                result = self.add_one(info)
                if result is None:
                    self.logger.warning(
                        f"Item with hash_id {info.get('hash_id')} could not be added"
                    )
                else:
                    self.logger.info(
                        f"Item with hash_id {info.get('hash_id')} added to the database"
                    )
            session.commit()  # Commit at the end of the loop


    def delete_many(self,ids: list):
        """
        Delete multiple entries from the database by their ids.

        Args:
            ids (list): List of ids of the entries to delete.

        Returns:
            None
        """
        with Session(self.engine) as session:
            for id in ids:
                # Reuse delete_one function for each item
                result = self.delete_one(id)
                if result is None:
                    self.logger.warning(f"No item with hash_id {id} found for deletion")
                else:
                    self.logger.info(f"Item with hash_id {id} deleted from the database")
            session.commit()  # Commit at the end of the loop


    def read_all(self,query: dict = None):
        """
        Read all entries from the database, optionally filtered by a query.

        Args:
            query (dict, optional): Dictionary containing the query parameters. Defaults to None.

        Returns:
            list: List of matching entries from the database.
        """
        with Session(self.engine) as session:
            statement = select(Sources)
            if query:
                statement = statement.where(
                    *[getattr(Sources, key) == value for key, value in query.items()]
                )
            sources = session.exec(statement).all()
            return sources


    def delete_all(self,):
        """
        Delete all entries from the database.

        Returns:
            None
        """
        with Session(self.engine) as session:
            session.exec(Sources).delete()
            session.commit()
            self.logger.info("All items deleted from the database")