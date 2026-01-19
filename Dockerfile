# Start from the official Ubuntu Bionic (18.04 LTS) image
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

# Copy all files except generate_Data and train_SM directories
#COPY --exclude=generate_Data --exclude=train_SM . /home/repo

COPY env_311.yml /usr/bin/environment.yml
COPY entrypoint_311.sh /usr/bin/entrypoint.sh

RUN useradd -ms /bin/bash -u 0 -o user

# Install any extra things we might need
RUN apt-get update \
        && apt-get install -y \
                vim \
                ssh \
                sudo \
                nano \
                locate \
                wget \
                software-properties-common ;\
                rm -rf /var/lib/apt/lists/*


# Install OpenFOAM v8 (without ParaView)
# including configuring for use by user=foam
# plus an extra environment variable to make OpenMPI play nice
RUN sh -c "wget -O - http://dl.openfoam.org/gpg.key | apt-key add -" && \
        add-apt-repository http://dl.openfoam.org/ubuntu && \
        apt-get update && \
        apt-get install -y --no-install-recommends openfoam8 && \
        rm -rf /var/lib/apt/lists/* && \
        echo "source /opt/openfoam8/etc/bashrc" >> ~user/.bashrc && \
        echo "export OMPI_MCA_btl_vader_single_copy_mechanism=none" >>  ~user/.bashrc

# Install miniconda
ENV CONDA_DIR=/opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O /tmp/miniconda.sh && \
        /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
        rm /tmp/miniconda.sh && \
        $CONDA_DIR/bin/conda clean -afy

# Creating python virtual environment
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda env create -f /usr/bin/environment.yml

# Giving enough permissions to the user
RUN mkdir -p /home/repo

# Copy repository contents to /home/repo
COPY --exclude=other_solvers_2d . /home/repo

WORKDIR /home/repo/

# set the default container user to foam
USER user

# The solvers will be installed in the entrypoint when running this image in a container
ENTRYPOINT ["/usr/bin/entrypoint.sh"]

