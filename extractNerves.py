from NervesFile import Nerves
from configFile import Config
configModel = Config()
dataModel = Nerves(configModel)
#dataModel.extractNerves()
#dataModel.extractNotNerves()
dataModel.trainingDataToPickle(configModel.maxWidth, configModel.maxHeight, 'nerves')
#dataModel.trainingDataToPickle(configModel.maxWidth, configModel.maxHeight, 'nerves', 'png')
#dataModel.generateMoreNervesContoursImages()
