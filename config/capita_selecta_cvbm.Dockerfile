FROM leandatascience/jupyterlab-ds-ml:latest
ENV MAIN_PATH=/Users/beerendgerats/Documents/University\ of\ Twente/Computer\ Science/Capita\ Selecta/capita_selecta_cvbm
ENV LIBS_PATH=${MAIN_PATH}/libs
ENV CONFIG_PATH=${MAIN_PATH}/config
ENV NOTEBOOK_PATH=${MAIN_PATH}/notebooks

EXPOSE 8888

CMD cd ${MAIN_PATH} && sh config/run_jupyter.sh
