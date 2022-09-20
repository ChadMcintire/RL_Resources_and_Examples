FROM ubuntu:20.04
# Configuration for displaying with X11
RUN mkdir /etc/sudoers.d
ARG USERNAME=rl
RUN useradd -m $USERNAME && \
echo "$USERNAME:$USERNAME" | chpasswd && \
usermod --shell /bin/bash $USERNAME && \
usermod -aG sudo $USERNAME && \
echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
chmod 0440 /etc/sudoers.d/$USERNAME && \
usermod --uid 1000 $USERNAME && \
groupmod --gid 1000 $USERNAME

#ENV LD_LIBRARY_PATH=/usr/lib/nvidia

ENV LD_LIBRARY_PATH=/home/$USERNAME/.mujoco/mjpro150/bin

#
#install directions for mujoco

RUN apt-get update -y && apt-get install -y libosmesa6-dev wget unzip python3-pip patchelf libasound2-dev 
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN mkdir installations && chown rl installations
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rl/.mujoco/mjpro150/bin 
RUN pip3 install gym==0.13.1 
RUN pip3 install beepy 
RUN cd installations/ && \
    wget https://www.roboti.us/download/mjpro150_linux.zip  && \
    unzip mjpro150_linux.zip  && \
    mkdir /home/rl/.mujoco && chown rl /home/rl/.mujoco && \
    mv mjpro150/ /home/rl/.mujoco/ && \
    wget https://www.roboti.us/file/mjkey.txt && \
    mv mjkey.txt /home/rl/.mujoco/ 


RUN pip install torchinfo && \
pip3 install tensorboard

COPY ./check_points ./

COPY ./main.py model.py sac.py training.py utils.py replay_memory.py execute_trained_model.py ./
# Set the 


#RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y locales  && \
#sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
#    dpkg-reconfigure --frontend=noninteractive locales && \
#    update-locale LANG=en_US.UTF-8 

#ENV LANG en_US.UTF-8 
 
#ENV LC_ALL en_US.UTF-8

#RUN apt-get install -y locales
#RUN apt-get install -y language-pack-en
#ENV LANG en_US.UTF-8 
#ENV LANGUAGE en_US:en 
#ENV LC_ALL en_US.UTF-8
#RUN python3 -c "print('UTF8 works nice! 👌')"

#RUN  apt install nvidia-driver-510 -y

USER $USERNAME

RUN pip3 install -U 'mujoco-py<1.50.2,>=1.50.1' 


CMD ["python3", "main.py", "--env-name", "Humanoid-v2", "--alpha", "0.1", "--render",  "--no-train",  "--cuda", "--check_pt_name", "check_points/alt-mod-no-auto/sac_cp_tuning=True_Gaussian_Humanoid-v2_0.05_990528"]