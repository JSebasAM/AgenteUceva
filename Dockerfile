FROM ubuntu:latest
LABEL authors="jseba"

ENTRYPOINT ["top", "-b"]