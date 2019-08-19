#from pip._internal import main as pipmain
#pipmain(['install','scipy'])

import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import numpy as np
import math
import keras
from keras.models import Model, load_model
from keras import backend as K
from scipy import ndimage
from skimage.exposure import rescale_intensity
import cv2
from skimage.transform import resize
#
# CNNSeg
#
#11
class CNNSeg(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "CNNSeg" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Examples"]
    self.parent.dependencies = []
    self.parent.contributors = ["John Doe (AnyWare Corp.)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
It performs a simple thresholding on the input volume and optionally captures a screenshot.
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""" # replace with organization, grant and thanks.

#
# CNNSegWidget
#

class CNNSegWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
  def inner(y_true, y_pred):
    """Computes the average exponential log Dice coefficients as the loss function.
    :param y_true: one-hot tensor multiplied by label weights, (batch size, number of pixels, number of labels).
    :param y_pred: softmax probabilities, same shape as y_true. Each probability serves as a partial volume.
    :return: average exponential log Dice coefficient.
    """
    exp = 1.0
    smooth = 1.0
    y_true = K.cast(K.not_equal(y_true, 0), K.floatx())  # Change to binary
    intersection = K.sum(y_true * y_pred, axis=1)  # (batch size, number of labels)
    union = K.sum(y_true + y_pred, axis=1)  # (batch size, number of labels)
    dice = (2. * intersection + smooth) / (union + smooth)
    #dice = generalized_dice(y_true, y_pred, exp)
    dice = K.clip(dice, K.epsilon(), 1 - K.epsilon())  # As log is used
    dice = K.pow(-K.log(dice), exp)
    if K.ndim(dice) == 2:
       dice = K.mean(dice, axis=-1)
    return dice

  ###################### Load Keras Model ###################
  segmentation_model = load_model('/Users/junyuchen/CNNSeg/CNNSeg/CNNSeg/'+'model_subnet_conc60.h5', custom_objects={'inner': inner})

  def setup(self):

    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    inputCollapsibleButton = ctk.ctkCollapsibleButton()
    inputCollapsibleButton.text = "Inputs"
    self.layout.addWidget(inputCollapsibleButton)

    outputsCollapsibleButton = ctk.ctkCollapsibleButton()
    outputsCollapsibleButton.text = "Outputs"
    self.layout.addWidget(outputsCollapsibleButton)

    # Layout within the dummy collapsible button
    inputsFormLayout = qt.QFormLayout(inputCollapsibleButton)
    outputsFormLayout = qt.QFormLayout(outputsCollapsibleButton)
    
    #
    # input volume selector SPECT
    #
    self.inputSelector_spect = slicer.qMRMLNodeComboBox()
    self.inputSelector_spect.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputSelector_spect.selectNodeUponCreation = True
    self.inputSelector_spect.addEnabled = False
    self.inputSelector_spect.removeEnabled = False
    self.inputSelector_spect.noneEnabled = False
    self.inputSelector_spect.showHidden = False
    self.inputSelector_spect.showChildNodeTypes = False
    self.inputSelector_spect.setMRMLScene( slicer.mrmlScene )
    self.inputSelector_spect.setToolTip( "Pick SPECT image" )
    inputsFormLayout.addRow("SPECT Volume: ", self.inputSelector_spect)

    #
    # input volume selector
    #
    self.inputSelector_ct = slicer.qMRMLNodeComboBox()
    self.inputSelector_ct.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputSelector_ct.selectNodeUponCreation = True
    self.inputSelector_ct.addEnabled = False
    self.inputSelector_ct.removeEnabled = False
    self.inputSelector_ct.noneEnabled = False
    self.inputSelector_ct.showHidden = False
    self.inputSelector_ct.showChildNodeTypes = False
    self.inputSelector_ct.setMRMLScene( slicer.mrmlScene )
    self.inputSelector_ct.setToolTip( "Pick CT image" )
    inputsFormLayout.addRow("CT Volume: ", self.inputSelector_ct)

    #
    # output clustering volume selector
    #
    self.outputLesClusterSelector = slicer.qMRMLNodeComboBox()
    self.outputLesClusterSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
    self.outputLesClusterSelector.selectNodeUponCreation = True
    self.outputLesClusterSelector.addEnabled = True
    self.outputLesClusterSelector.removeEnabled = True
    self.outputLesClusterSelector.noneEnabled = True
    self.outputLesClusterSelector.showHidden = False
    self.outputLesClusterSelector.showChildNodeTypes = False
    self.outputLesClusterSelector.setMRMLScene( slicer.mrmlScene )
    self.outputLesClusterSelector.setToolTip( "Pick the output to the algorithm." )
    outputsFormLayout.addRow("Output Lesion Label Map: ", self.outputLesClusterSelector)

    #
    # output volume selector
    #
    self.outputBoneClusterSelector = slicer.qMRMLNodeComboBox()
    self.outputBoneClusterSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
    self.outputBoneClusterSelector.selectNodeUponCreation = True
    self.outputBoneClusterSelector.addEnabled = True
    self.outputBoneClusterSelector.removeEnabled = True
    self.outputBoneClusterSelector.noneEnabled = True
    self.outputBoneClusterSelector.showHidden = False
    self.outputBoneClusterSelector.showChildNodeTypes = False
    self.outputBoneClusterSelector.setMRMLScene( slicer.mrmlScene )
    self.outputBoneClusterSelector.setToolTip( "Pick the output to the algorithm." )
    outputsFormLayout.addRow("Output Bone Label Map: ", self.outputBoneClusterSelector)

    #
    # output Les seg selector
    #
    self.segLesSelector = slicer.qMRMLNodeComboBox()
    self.segLesSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
    self.segLesSelector.selectNodeUponCreation = True
    self.segLesSelector.addEnabled = True
    self.segLesSelector.removeEnabled = True
    self.segLesSelector.noneEnabled = True
    self.segLesSelector.showHidden = False
    self.segLesSelector.showChildNodeTypes = False
    self.segLesSelector.setMRMLScene( slicer.mrmlScene )
    self.segLesSelector.setToolTip( "Pick the output to the algorithm." )
    outputsFormLayout.addRow("Output Lesion Segmentation: ", self.segLesSelector)

    #
    # output Bone seg selector
    #
    self.segBoneSelector = slicer.qMRMLNodeComboBox()
    self.segBoneSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
    self.segBoneSelector.selectNodeUponCreation = True
    self.segBoneSelector.addEnabled = True
    self.segBoneSelector.removeEnabled = True
    self.segBoneSelector.noneEnabled = True
    self.segBoneSelector.showHidden = False
    self.segBoneSelector.showChildNodeTypes = False
    self.segBoneSelector.setMRMLScene( slicer.mrmlScene )
    self.segBoneSelector.setToolTip( "Pick the output to the algorithm." )
    outputsFormLayout.addRow("Output Bone Segmentation: ", self.segBoneSelector)

    #
    # check box to trigger taking screen shots for later use in tutorials
    #
    self.enableScreenshotsFlagCheckBox = qt.QCheckBox()
    self.enableScreenshotsFlagCheckBox.checked = 0
    self.enableScreenshotsFlagCheckBox.setToolTip("If checked, take screen shots for tutorials. Use Save Data to write them to disk.")
    inputsFormLayout.addRow("Enable Screenshots", self.enableScreenshotsFlagCheckBox)

    self.segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    self.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    self.segmentEditorWidget.rotateSliceViewsToSegmentation()

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = False
    self.layout.addWidget(self.applyButton)

    #
    # Advanced Button
    #
    advancedCollapsibleButton = ctk.ctkCollapsibleButton()
    advancedCollapsibleButton.text = "Advanced"
    self.layout.addWidget(advancedCollapsibleButton)
    advancedFormLayout = qt.QFormLayout(advancedCollapsibleButton)
    

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.inputSelector_spect.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.inputSelector_ct.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.outputBoneClusterSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.outputLesClusterSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.segLesSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.segBoneSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()

  def cleanup(self):
    pass

  def onSelect(self):
    self.applyButton.enabled = self.inputSelector_spect.currentNode() and self.inputSelector_ct.currentNode() and self.outputBoneClusterSelector.currentNode() and self.segLesSelector.currentNode() and self.segBoneSelector.currentNode() and self.outputLesClusterSelector.currentNode()

  def onApplyButton(self):
    logic = CNNSegLogic()
    enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
    logic.run(self.inputSelector_spect.currentNode(), self.inputSelector_ct.currentNode(), self.outputBoneClusterSelector.currentNode(), self.outputLesClusterSelector.currentNode(), self.segLesSelector.currentNode(), self.segBoneSelector.currentNode(), enableScreenshotsFlag)

#
# testLogic
#

#
# CNNSegLogic
#

class CNNSegLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def hasImageData(self,volumeNode):
    """This is an example logic method that
    returns true if the passed in volume
    node has valid image data
    """
    if not volumeNode:
      logging.debug('hasImageData failed: no volume node')
      return False
    if volumeNode.GetImageData() is None:
      logging.debug('hasImageData failed: no image data in volume node')
      return False
    return True

  def isValidInputOutputData(self, inputVolumeNode_spect, inputVolumeNode_ct , outputBoneClusterNode, outputLesClusterNode, outputLesSegmentationNode, outputBoneSegmentationNode):
    """Validates if the output is not the same as input
    """
    if not inputVolumeNode_spect:
      logging.debug('isValidInputOutputData failed: no input volume node defined')
      return False
    if not inputVolumeNode_ct:
      logging.debug('isValidInputOutputData failed: no input volume node defined')
      return False
    if not outputBoneClusterNode:
      logging.debug('isValidInputOutputData failed: no output volume node defined')
      return False
    if not outputLesSegmentationNode:
      logging.debug('isValidInputOutputData failed: no output segmentation node defined')
      return False
    if not outputBoneSegmentationNode:
      logging.debug('isValidInputOutputData failed: no output segmentation node defined')
      return False
    if not outputLesClusterNode:
      logging.debug('isValidInputOutputData failed: no output cluster node defined')
      return False
    #if inputVolumeNode_spect.GetID()==outputLesVolumeNode.GetID():
     # logging.debug('isValidInputOutputData failed: input and output volume is the same. Create a new volume for output to avoid this error.')
      #return False
    return True

  def takeScreenshot(self,name,description,type=-1):
    # show the message even if not taking a screen shot
    slicer.util.delayDisplay('Take screenshot: '+description+'.\nResult is available in the Annotations module.', 3000)

    lm = slicer.app.layoutManager()
    # switch on the type to get the requested window
    widget = 0
    if type == slicer.qMRMLScreenShotDialog.FullLayout:
      # full layout
      widget = lm.viewport()
    elif type == slicer.qMRMLScreenShotDialog.ThreeD:
      # just the 3D window
      widget = lm.threeDWidget(0).threeDView()
    elif type == slicer.qMRMLScreenShotDialog.Red:
      # red slice window
      widget = lm.sliceWidget("Red")
    elif type == slicer.qMRMLScreenShotDialog.Yellow:
      # yellow slice window
      widget = lm.sliceWidget("Yellow")
    elif type == slicer.qMRMLScreenShotDialog.Green:
      # green slice window
      widget = lm.sliceWidget("Green")
    else:
      # default to using the full window
      widget = slicer.util.mainWindow()
      # reset the type so that the node is set correctly
      type = slicer.qMRMLScreenShotDialog.FullLayout

    # grab and convert to vtk image data
    qimage = ctk.ctkWidgetsUtils.grabWidget(widget)
    imageData = vtk.vtkImageData()
    slicer.qMRMLUtils().qImageToVtkImageData(qimage,imageData)

    annotationLogic = slicer.modules.annotations.logic()
    annotationLogic.CreateSnapShot(name, description, type, 1, imageData)

  def preprocessing(self, patientData, attnData):
    patientData = np.einsum('kli->lki', patientData)
    attnData = np.einsum('kli->lki', attnData)

    slices = patientData.shape[1]
    processed_img = np.zeros((slices,192,192*2))

    patientData = rescale_intensity(patientData, out_range=(0, 255))
    attnData = rescale_intensity(attnData, out_range=(0, 255))

    for i in range(slices):
      patient = patientData[:,i,:]
      attn    = attnData[:,i,:]
      attn_tmp= np.array(attn)

      #patient = rescale_intensity(patient, out_range=(0, 255))
      #attn_tmp = rescale_intensity(attn_tmp, out_range=(0, 255))
      attn = rescale_intensity(attn.astype(np.float), out_range=(0, 255))

      ret2,th2 = cv2.threshold(attn_tmp.astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
      attn_m = th2
      attn_m = ndimage.binary_fill_holes(attn_m).astype(int)
      attn_m = ndimage.binary_erosion(attn_m).astype(attn_m.dtype)
      attn_m = ndimage.binary_dilation(attn_m).astype(attn_m.dtype)
      attn[attn_m == 0] = 0

      #print(attn.shape)
      #print(patient.shape)
      attn = resize(attn, (192, 192),anti_aliasing=False, preserve_range=True)
      patient = resize(patient, (192, 192),anti_aliasing=False, preserve_range=True)
      #attn = rescale_intensity(attn, out_range=(0, 255))
      #patient = rescale_intensity(patient, out_range=(0, 255))

      #if i == int(slices/2): plt.figure(1); plt.imshow(attn,cmap='gray'); plt.show()
      #if i == int(slices/2): plt.figure(1); plt.imshow(patient,cmap='gray'); plt.show()
      #output_name = 'img.' + str(i)
      processed_img[i,:,:] = np.concatenate((attn, patient), axis=1)
    return processed_img

  def get_segmentation(self, img1, img2):
    output = CNNSegWidget.segmentation_model.predict([img1, img2])
    try:
      output = np.nan_to_num(output)
      output[output>=0.5]=1
      output[output<0.5]=0
    except:
      output = np.zeros_like(output)
    return output


  def run(self, inputVolume_spect, inputVolume_ct,outputBoneClusterVolume, outputLesClusterVolume,outputLesSegmentation, outputBoneSegmentation, enableScreenshots=0):
    
    
    """
    Run the actual algorithm
    """
    outputLesSegmentation.GetSegmentation().RemoveAllSegments()
    outputBoneSegmentation.GetSegmentation().RemoveAllSegments()
    if not self.isValidInputOutputData(inputVolume_spect, inputVolume_ct, outputBoneClusterVolume, outputLesClusterVolume, outputLesSegmentation, outputBoneSegmentation):
      slicer.util.errorDisplay('Input volume is the same as output volume. Choose a different output volume.')
      return False

    logging.info('Processing started')

     # Compute the thresholded output volume using the Threshold Scalar Volume CLI module
    cliParams = {'InputVolume': inputVolume_ct.GetID(), 'OutputVolume': outputBoneClusterVolume.GetID(), 'ThresholdValue' : 0.2, 'ThresholdType' : 'Above'}
    cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True)
    cliParams = {'InputVolume': inputVolume_ct.GetID(), 'OutputVolume': outputLesClusterVolume.GetID(), 'ThresholdValue' : 0.2, 'ThresholdType' : 'Above'}
    cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True)

    # Capture screenshot
    if enableScreenshots:
      self.takeScreenshot('testTest-Start','MyScreenshot',-1)

    # convert volume to nd array
    spect_img = list(slicer.util.arrayFromVolume(inputVolume_spect))
    spect_img = np.asarray(spect_img)

    # convert volume to nd array
    ct_img = list(slicer.util.arrayFromVolume(inputVolume_ct))
    ct_img = np.asarray(ct_img)
    
    spect_org_shape = spect_img.shape
    ct_org_shape = ct_img.shape

    vol_size = inputVolume_spect.GetImageData().GetDimensions()
    vol_size = np.asarray(vol_size)
    vol_center = vol_size/2

    #print('spect size: '+str(np.shape(spect_img)))
    #print('ct size: '+str(np.shape(ct_img)))

    image_test = self.preprocessing(spect_img, ct_img)
    image_test  = image_test.reshape(image_test.shape[0], 192, 192*2, 1)
    bone_label = np.zeros((spect_org_shape[1],image_test.shape[0],spect_org_shape[2]))
    les_label = np.zeros((spect_org_shape[1],image_test.shape[0],spect_org_shape[2]))
    for img_i in range(image_test.shape[0]):
      #print(img_i)
      img = image_test[img_i,:,:,:]
      img = img.reshape(1,192,192*2,1)
      imgCT = img[:, :, 0:192, :]
      imgSPECT = img[:, :, 192:192 * 2, :]
      orig_seg = self.get_segmentation(imgSPECT, imgCT)
      seg_out = orig_seg.reshape(len(imgSPECT), 192, 192, 3)
      #print(seg_out.shape)
      bone_seg =  seg_out[0,:,:,1] + seg_out[0,:,:,2]
      bone_seg.reshape(192,192)
      bone_seg = resize(bone_seg, (spect_org_shape[1], spect_org_shape[2]),order = 0, anti_aliasing=False)
      bone_label[:,img_i,:] = bone_seg
      les_seg = seg_out[0,:,:,1]
      les_seg.reshape(192,192)
      les_seg = resize(les_seg, (spect_org_shape[1], spect_org_shape[2]),order = 0, anti_aliasing=False)
      les_label[:,img_i,:] = les_seg
    bone_label = ndimage.binary_dilation(bone_label).astype(bone_label.dtype)
    les_label = ndimage.binary_dilation(les_label).astype(les_label.dtype)
    bone_label = np.einsum('lki->kli', bone_label)
    les_label = np.einsum('lki->kli', les_label)
    les_label = les_label * 2
    slicer.util.updateVolumeFromArray(outputLesClusterVolume,les_label) # clustering results
    slicer.util.updateVolumeFromArray(outputBoneClusterVolume,bone_label)
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(outputLesClusterVolume, outputLesSegmentation)
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(outputBoneClusterVolume, outputBoneSegmentation) 
    

    logging.info('Processing completed')


    return True

class CNNSegTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_CNNSeg1()

  def test_CNNSeg1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    import urllib
    downloads = (
        ('http://slicer.kitware.com/midas3/download?items=5767', 'FA.nrrd', slicer.util.loadVolume),
        )

    for url,name,loader in downloads:
      filePath = slicer.app.temporaryPath + '/' + name
      if not os.path.exists(filePath) or os.stat(filePath).st_size == 0:
        logging.info('Requesting download %s from %s...\n' % (name, url))
        urllib.urlretrieve(url, filePath)
      if loader:
        logging.info('Loading %s...' % (name,))
        loader(filePath)
    self.delayDisplay('Finished with download and loading')

    volumeNode = slicer.util.getNode(pattern="FA")
    logic = CNNSegLogic()
    self.assertIsNotNone( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')
