#from sqlalchemy import * #create_engine, Column
from sqlalchemy import create_engine, Column, Integer, String, Sequence, Float,PrimaryKeyConstraint, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, backref

from sqlalchemy.sql import *
#from db import PrimaryKeyConstraint
#engine = create_engine('sqlite:///demo.db')
import os
#from sqlalchemy import create_engine
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
DB_IP=os.getenv("DB_IP")


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
    DB_IP=os.getenv("DB_IP")

def connect_db():
    #_load_db_vars()
    # create db create_engine
    db = create_engine(f'postgresql://{DB_USER}:{DB_PASS}@{DB_IP}:{DB_PORT}/{DB_NAME}')
    return db

Base = declarative_base()

class PointsOfInterest(Base):
    __tablename__ = "points_of_interest"
    poi_id = Column(Integer, Sequence('poi_id_seq'), primary_key=True)
    name = Column(String)
    build_year = Column(String)
    demolished_year = Column(String)
    address = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    external_url = Column(String)
    image_url = Column(String)
    heritage_status = Column(String)
    current_use = Column(String)
    poi_type = Column(String)
    source = Column(String)
    details = Column(String)
    #Defining One to Many relationships with the relationship function on the Parent Table
    styles = relationship('ArchitecturalStyles', backref = 'points_of_interest',lazy=True,cascade="all, delete-orphan")
    architects = relationship('Architects', backref = 'points_of_interest', lazy=True,cascade="all, delete-orphan")
    categories = relationship('POICategories', backref = 'points_of_interest', lazy=True,cascade="all, delete-orphan")

class ArchitecturalStyles(Base):
    __tablename__="architectural_styles"
    __table_args__ = (
        PrimaryKeyConstraint('poi_id', 'style'),
    )
    poi_id =Column(Integer,ForeignKey('points_of_interest.poi_id'))
    #Defining the Foreign Key on the Child Table
    style = Column(String)

class Architects(Base):
    __tablename__="architects"
    __table_args__ = (
        PrimaryKeyConstraint('poi_id', 'architect_name'),
    )
    poi_id= Column(Integer,ForeignKey('points_of_interest.poi_id'))
    architect_name = Column(String)

class POICategories(Base):
    __tablename__="poi_categories"
    __table_args__ = (
        PrimaryKeyConstraint('poi_id', 'category'),
    )
    poi_id =Column(Integer,ForeignKey('points_of_interest.poi_id'))
    category = Column(String)


engine = connect_db()
PointsOfInterest.__table__.create(bind=engine, checkfirst=True)
ArchitecturalStyles.__table__.create(bind=engine, checkfirst=True)
Architects.__table__.create(bind=engine, checkfirst=True)
POICategories.__table__.create(bind=engine, checkfirst=True)
