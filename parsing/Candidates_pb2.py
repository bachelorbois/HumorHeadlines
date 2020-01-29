# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: Candidates.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='Candidates.proto',
  package='Candidates',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x10\x43\x61ndidates.proto\x12\nCandidates\"\xc4\x03\n\x13\x43\x61ndidateCollection\x12>\n\ncandidates\x18\x01 \x03(\x0b\x32*.Candidates.CandidateCollection.Candidates\x1a\xec\x02\n\nCandidates\x12@\n\x03HL1\x18\x01 \x01(\x0b\x32\x33.Candidates.CandidateCollection.Candidates.Headline\x12@\n\x03HL2\x18\x02 \x01(\x0b\x32\x33.Candidates.CandidateCollection.Candidates.Headline\x12?\n\x05label\x18\x03 \x01(\x0e\x32\x30.Candidates.CandidateCollection.Candidates.Label\x1am\n\x08Headline\x12\n\n\x02id\x18\x01 \x01(\r\x12\x10\n\x08sentence\x18\x02 \x03(\t\x12\x12\n\nword_index\x18\x03 \x01(\r\x12\x0c\n\x04\x65\x64it\x18\x04 \x01(\t\x12\x0e\n\x06grades\x18\x05 \x03(\r\x12\x11\n\tavg_grade\x18\x06 \x01(\x02\"*\n\x05Label\x12\t\n\x05\x45QUAL\x10\x00\x12\x06\n\x02H1\x10\x01\x12\x06\n\x02H2\x10\x02\x12\x06\n\x02NA\x10\x03\x62\x06proto3'
)



_CANDIDATECOLLECTION_CANDIDATES_LABEL = _descriptor.EnumDescriptor(
  name='Label',
  full_name='Candidates.CandidateCollection.Candidates.Label',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='EQUAL', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='H1', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='H2', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NA', index=3, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=443,
  serialized_end=485,
)
_sym_db.RegisterEnumDescriptor(_CANDIDATECOLLECTION_CANDIDATES_LABEL)


_CANDIDATECOLLECTION_CANDIDATES_HEADLINE = _descriptor.Descriptor(
  name='Headline',
  full_name='Candidates.CandidateCollection.Candidates.Headline',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='Candidates.CandidateCollection.Candidates.Headline.id', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sentence', full_name='Candidates.CandidateCollection.Candidates.Headline.sentence', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='word_index', full_name='Candidates.CandidateCollection.Candidates.Headline.word_index', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='edit', full_name='Candidates.CandidateCollection.Candidates.Headline.edit', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='grades', full_name='Candidates.CandidateCollection.Candidates.Headline.grades', index=4,
      number=5, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='avg_grade', full_name='Candidates.CandidateCollection.Candidates.Headline.avg_grade', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=332,
  serialized_end=441,
)

_CANDIDATECOLLECTION_CANDIDATES = _descriptor.Descriptor(
  name='Candidates',
  full_name='Candidates.CandidateCollection.Candidates',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='HL1', full_name='Candidates.CandidateCollection.Candidates.HL1', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='HL2', full_name='Candidates.CandidateCollection.Candidates.HL2', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='label', full_name='Candidates.CandidateCollection.Candidates.label', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_CANDIDATECOLLECTION_CANDIDATES_HEADLINE, ],
  enum_types=[
    _CANDIDATECOLLECTION_CANDIDATES_LABEL,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=121,
  serialized_end=485,
)

_CANDIDATECOLLECTION = _descriptor.Descriptor(
  name='CandidateCollection',
  full_name='Candidates.CandidateCollection',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='candidates', full_name='Candidates.CandidateCollection.candidates', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_CANDIDATECOLLECTION_CANDIDATES, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=33,
  serialized_end=485,
)

_CANDIDATECOLLECTION_CANDIDATES_HEADLINE.containing_type = _CANDIDATECOLLECTION_CANDIDATES
_CANDIDATECOLLECTION_CANDIDATES.fields_by_name['HL1'].message_type = _CANDIDATECOLLECTION_CANDIDATES_HEADLINE
_CANDIDATECOLLECTION_CANDIDATES.fields_by_name['HL2'].message_type = _CANDIDATECOLLECTION_CANDIDATES_HEADLINE
_CANDIDATECOLLECTION_CANDIDATES.fields_by_name['label'].enum_type = _CANDIDATECOLLECTION_CANDIDATES_LABEL
_CANDIDATECOLLECTION_CANDIDATES.containing_type = _CANDIDATECOLLECTION
_CANDIDATECOLLECTION_CANDIDATES_LABEL.containing_type = _CANDIDATECOLLECTION_CANDIDATES
_CANDIDATECOLLECTION.fields_by_name['candidates'].message_type = _CANDIDATECOLLECTION_CANDIDATES
DESCRIPTOR.message_types_by_name['CandidateCollection'] = _CANDIDATECOLLECTION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

CandidateCollection = _reflection.GeneratedProtocolMessageType('CandidateCollection', (_message.Message,), {

  'Candidates' : _reflection.GeneratedProtocolMessageType('Candidates', (_message.Message,), {

    'Headline' : _reflection.GeneratedProtocolMessageType('Headline', (_message.Message,), {
      'DESCRIPTOR' : _CANDIDATECOLLECTION_CANDIDATES_HEADLINE,
      '__module__' : 'Candidates_pb2'
      # @@protoc_insertion_point(class_scope:Candidates.CandidateCollection.Candidates.Headline)
      })
    ,
    'DESCRIPTOR' : _CANDIDATECOLLECTION_CANDIDATES,
    '__module__' : 'Candidates_pb2'
    # @@protoc_insertion_point(class_scope:Candidates.CandidateCollection.Candidates)
    })
  ,
  'DESCRIPTOR' : _CANDIDATECOLLECTION,
  '__module__' : 'Candidates_pb2'
  # @@protoc_insertion_point(class_scope:Candidates.CandidateCollection)
  })
_sym_db.RegisterMessage(CandidateCollection)
_sym_db.RegisterMessage(CandidateCollection.Candidates)
_sym_db.RegisterMessage(CandidateCollection.Candidates.Headline)


# @@protoc_insertion_point(module_scope)
