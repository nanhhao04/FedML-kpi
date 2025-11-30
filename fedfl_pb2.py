# -*- coding: utf-8 -*-
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    29,
    0,
    '',
    'fedfl.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0b\x66\x65\x64\x66l.proto\"5\n\x0bJoinRequest\x12\x11\n\tclient_id\x18\x01 \x01(\x05\x12\x13\n\x0bnum_samples\x18\x02 \x01(\x05\"\x92\x01\n\x12LocalUpdateRequest\x12\x11\n\tclient_id\x18\x01 \x01(\x05\x12\x1a\n\x12serialized_weights\x18\x02 \x01(\x0c\x12\x13\n\x0bnum_samples\x18\x03 \x01(\x05\x12\x12\n\ntrain_loss\x18\x04 \x01(\x02\x12\x11\n\ttrain_mae\x18\x05 \x01(\x02\x12\x11\n\ttrain_mse\x18\x06 \x01(\x02\"d\n\x0fWeightsResponse\x12\x1a\n\x12serialized_weights\x18\x01 \x01(\x0c\x12\x10\n\x08round_id\x18\x02 \x01(\x05\x12\x10\n\x08is_final\x18\x03 \x01(\x08\x12\x11\n\twait_join\x18\x04 \x01(\x08\x32\x7f\n\x11\x46\x65\x64\x65rationService\x12\x30\n\x0eJoinFederation\x12\x0c.JoinRequest\x1a\x10.WeightsResponse\x12\x38\n\x0fSendLocalUpdate\x12\x13.LocalUpdateRequest\x1a\x10.WeightsResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'fedfl_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_JOINREQUEST']._serialized_start=15
  _globals['_JOINREQUEST']._serialized_end=68
  _globals['_LOCALUPDATEREQUEST']._serialized_start=71
  _globals['_LOCALUPDATEREQUEST']._serialized_end=217
  _globals['_WEIGHTSRESPONSE']._serialized_start=219
  _globals['_WEIGHTSRESPONSE']._serialized_end=319
  _globals['_FEDERATIONSERVICE']._serialized_start=321
  _globals['_FEDERATIONSERVICE']._serialized_end=448
# @@protoc_insertion_point(module_scope)
