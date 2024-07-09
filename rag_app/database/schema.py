from sqlmodel import SQLModel, Field
from typing import Optional
import datetime

class Sources(SQLModel, table=True):
    """
    Database schema for the Sources table.

    Attributes:
        id (Optional[int]): The primary key for the table.
        url (str): The URL of the source.
        title (Optional[str]): The title of the source.
        hash_id (str): A unique identifier for the source.
        created_at (float): Timestamp indicating when the entry was created.
        summary (str): A summary of the source content.
        embedded (bool): Flag indicating whether the source is embedded.
        session_id (str): A unique identifier for the session when the entry was added.
        session_date_time (str): The timestamp when the session was created.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    url: str = Field()
    title: Optional[str] = Field(default="NA", unique=False)
    hash_id: str = Field(unique=True)
    created_at: float = Field(default=datetime.datetime.now().timestamp())
    summary: str = Field(default="")
    embedded: bool = Field(default=False)
    session_id: str = Field(default="")
    session_date_time: str = Field(default="")

    __table_args__ = {"extend_existing": True}
