# hash:sha256:9fc7d46102b301952f2e86740f5e5f0f258cee6ac29cdbe6dee1e36d94768566
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -a
ENV PATH /opt/conda/envs/magnets/bin:$PATH