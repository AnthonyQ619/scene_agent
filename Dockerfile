FROM kunalg106/cuda129
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create --name autosfm -y python=3.11
RUN sudo apt update && sudo apt install -y libgl1

RUN /opt/conda/envs/autosfm/bin/python -m pip install pycolmap-cuda12
RUN /opt/conda/envs/autosfm/bin/python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
RUN /opt/conda/envs/autosfm/bin/python -m pip install scipy open3d kornia piexif omegaconf pytransform3d sceneprogllm sceneprogsyn
RUN git clone https://github.com/facebookresearch/map-anything.git && cd map-anything && /opt/conda/envs/autosfm/bin/python -m pip install -e .
RUN git clone https://github.com/cvg/LightGlue.git && cd LightGlue && /opt/conda/envs/autosfm/bin/python -m pip install -e .
# RUN git clone https://github.com/AnthonyQ619/scene_agent.git
# RUN cd scene_agent/breadth_agent/src && /opt/conda/envs/autosfm/bin/python -m pip install -e .
# RUN cd scene_agent/breadth_agent/src/modules/models/sfm_models && /opt/conda/envs/autosfm/bin/python -m pip install .
