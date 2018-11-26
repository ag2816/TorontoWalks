import os
from sqlalchemy import create_engine
from dotenv import load_dotenv, find_dotenv

# USAGE: from create_db import connect_db, create_db_tables,test_database
# db=connect_db() --> establish connection
# db=create_db_tables(db)--> create database tables

load_dotenv(find_dotenv())
# load environment variables
SECRET_KEY = os.getenv("SECRET_KEY")
DB_USER=os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DEBUG= os.getenv("DEBUG")
DB_NAME=os.getenv("DB_NAME")
DB_SERVICE=os.getenv("DB_SERVICE")
DB_PORT=os.getenv("DB_PORT")


def _load_db_vars():
    load_dotenv(find_dotenv())
    # load environment variables
    SECRET_KEY = os.getenv("SECRET_KEY")
    DB_USER=os.getenv("DB_USER")
    DB_PASS = os.getenv("DB_PASS")
    DEBUG= os.getenv("DEBUG")
    DB_NAME=os.getenv("DB_NAME")
    DB_SERVICE=os.getenv("DB_SERVICE")
    DB_PORT=os.getenv("DB_PORT")


def connect_db():
    #_load_db_vars()
    # create db create_engine
    db = create_engine(f'postgresql://{DB_USER}:{DB_PASS}@192.168.99.101:{DB_PORT}/{DB_NAME}')
    return db

def create_db_tables(db):

    # create main table
    db.execute("""
    CREATE TABLE IF NOT EXISTS points_of_interest (poi_id BIGSERIAL PRIMARY KEY,
        name text,
        build_year text, demolished_year text,
        address text, latitude float, longitude float,
        source text, external_url text, details text,
        image_url text, heritage_status text, current_use text,
        poi_type text)
    """)

    # create architectural styles TABLE
    db.execute("""
    CREATE TABLE IF NOT EXISTS architectural_styles (poi_id int,
        style text
    )
    """)

    # create architects TABLE
    db.execute("""
    CREATE TABLE IF NOT EXISTS architects (poi_id int ,
        architect_name text
    )
    """)

    # create categories TABLE
    db.execute("""
    CREATE TABLE IF NOT EXISTS poi_categories (poi_id int,
        category text
    )
    """)

    return db
