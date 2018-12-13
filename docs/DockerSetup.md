# Notes on Docker Setup

Followed process outlined in https://www.patricksoftwareblog.com/using-docker-for-flask-application-development-not-just-production/

## Docker-compose.yml

Docker-compose connects different containers together to create an application

* top line is version of docker-compose to use (version 3 is latest)
* then define services - 1 service = 1 docker container
* build: ./web --> location of the docker file (at the root of the web subdirectory)
* Service 1: web / app
  * builds using docker file at root of web sub directory
  * Docker file
    * creates non-root user (flask)
    * pip installs all modules listed in requirements.txt file 
    * copies in source code
* Service 2: nginx
  * build: points to docker file at root of nginx directory
  * it acts as proxy server; receives HTTP requests and forwards them to Python app
  * Docker File: This is a very simple dockerfile that takes uses the latest Nginx docker image. It then removes the default configuration and adds the torontowalks configuration.
  * uses gunicorn 
  * The torontowalks.conf file is a simple Nginx configuration file which listens for traffic on port 80 (HTTP). It then passes on the data to uWSGI (hence `location /`). We then pass the HTTP request to another Docker container called `flask` on port 8080.

* Service 3: postgreSQL database
  * This service doesn’t have its own docker file but instead uses the official Postgres image
  * stores data on separate data volume
  * For the postgres container, the only change that is made is to allow access to port 5432 by the host machine instead of just other services

## Startup

* switch from For the web application container, the web server is being switched from Gunicorn (used in production) to the Flask development server
* http://ip_of_docker_machine:5000.

```
docker-compose build
# in DEV: (uses override)
docker-compose up -d
# in PRD (skips override)
docker-compose -f docker-compose.yml up -d

```

LESSONS LEARNED

* all files and directories need to be copied into the docker environment
* including log files, .env file (error on DP.PORT because unable to to find and read .env file)
* html file -- store all in templates directory and call render_template()
* if getting 

# Move to Digital Ocean

https://realpython.com/dockerizing-flask-with-compose-and-machine-from-localhost-to-the-cloud/

```
 docker-machine create \
-d digitalocean \
--digitalocean-access-token=ADD_YOUR_TOKEN_HERE \
torontowalks-prod
```



* the above actually creates the a new droplet in Digitalocean for you

* set it as the active machine and load in shell

* ```
  eval "$(docker-machine env torontowalks-prod)"
  docker-compose build
  $ docker-compose up -d
  ```

issues

* 502 Bad Gateway 
  * docker-compose logs nginx
  * 6#6: *17 connect() failed (113: No route to host) 
  * solution is to connect to port 8000 http://68.183.55.22:8000/page
  * also fix db IP in .env
