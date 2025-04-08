FROM pandoc/latex:3.6.4.0-ubuntu
RUN apt-get update && \
    apt-get upgrade -y && \
    tlmgr update --self --all && \
    tlmgr install collection-langjapanese
