from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class VolumeDriver():
    def __init__(self):
        self.devices = AudioUtilities.GetSpeakers()
        self.interface = self.devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(self.interface, POINTER(IAudioEndpointVolume))

    # def GetMute(self):
    #     self.volume.GetMute()
    #
    def GetMasterVolumeLevel(self):
        return self.volume.GetMasterVolumeLevel()

    def GetVolumeRange(self):
        return self.volume.GetVolumeRange()

    def SetMasterVolumeLevel(self, level=-30.0):
        self.volume.SetMasterVolumeLevel(level, None)
