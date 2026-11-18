import onnxruntime as ort
import onnxruntime_qnn as qnn_ep


ep_lib_path = qnn_ep.get_library_path()
lib_registration_name = "QNNExecutionProvider"
ort.register_execution_provider_library(lib_registration_name, ep_lib_path)

all_ep_devices = ort.get_ep_devices()
selected_ep_devices = [ep_device for ep_device in all_ep_devices if ep_device.ep_name == lib_registration_name and ep_device.device.type == ort.OrtHardwareDeviceType.NPU]

ep_options = { 'backend_path': qnn_ep.get_qnn_htp_path() }
sess_options = ort.SessionOptions()
sess_options.add_provider_for_devices(selected_ep_devices, ep_options)
