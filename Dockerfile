# Start from the official Ubuntu Bionic (18.04 LTS) image
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

# Copy all files except generate_Data and train_SM directories
#COPY --exclude=generate_Data --exclude=train_SM . /home/repo

COPY env_311.yml /usr/bin/environment.yml
COPY entrypoint_311.sh /usr/bin/entrypoint.sh

# Create a new user called user
RUN useradd -ms /bin/bash user
RUN usermod -aG sudo user

RUN mkdir -p /etc/sudoers.d


# Edit the /etc/sudoers file to allow the user to run sudo without a password prompt
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/user-nopasswd
RUN chmod 440 /etc/sudoers.d/user-nopasswd

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
#RUN conda activate python39 && conda install mpi4py


# Giving enough permissions to the user
RUN mkdir -p /home/repo
# If more permissions are required the user can chown itself (user) with sudo inside the container
RUN chown -R user:user /home/repo

WORKDIR /home/repo/
RUN sudo chmod 755 /usr/bin/entrypoint.sh

# set the default container user to foam
USER user

ENTRYPOINT ["/bin/bash","-l"]
# The solvers will be installed in the entrypoint when running this image in a container
#ENTRYPOINT ["/usr/bin/entrypoint.sh"]

