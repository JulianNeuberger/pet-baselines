from data import model
import numpy as np
from augment import base

# Author: Benedikt
class TrafoNullStep(base.AugmentationStep):

    name = "null"

    def do_augment(self, doc: model.Document) -> model.Document:
        return doc