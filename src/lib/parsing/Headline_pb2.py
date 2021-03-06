# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: Headline.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='Headline.proto',
  package='Headline',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x0eHeadline.proto\x12\x08Headline\"\xb6\x01\n\x12HeadlineCollection\x12\x31\n\x02HL\x18\x01 \x03(\x0b\x32%.Headline.HeadlineCollection.Headline\x1am\n\x08Headline\x12\n\n\x02id\x18\x01 \x01(\r\x12\x10\n\x08sentence\x18\x02 \x03(\t\x12\x12\n\nword_index\x18\x03 \x01(\r\x12\x0c\n\x04\x65\x64it\x18\x04 \x01(\t\x12\x0e\n\x06grades\x18\x05 \x03(\r\x12\x11\n\tavg_grade\x18\x06 \x01(\x02\x62\x06proto3'
)




_HEADLINECOLLECTION_HEADLINE = _descriptor.Descriptor(
  name='Headline',
  full_name='Headline.HeadlineCollection.Headline',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='Headline.HeadlineCollection.Headline.id', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sentence', full_name='Headline.HeadlineCollection.Headline.sentence', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='word_index', full_name='Headline.HeadlineCollection.Headline.word_index', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='edit', full_name='Headline.HeadlineCollection.Headline.edit', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='grades', full_name='Headline.HeadlineCollection.Headline.grades', index=4,
      number=5, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='avg_grade', full_name='Headline.HeadlineCollection.Headline.avg_grade', index=5,
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
  serialized_start=102,
  serialized_end=211,
)

_HEADLINECOLLECTION = _descriptor.Descriptor(
  name='HeadlineCollection',
  full_name='Headline.HeadlineCollection',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='HL', full_name='Headline.HeadlineCollection.HL', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_HEADLINECOLLECTION_HEADLINE, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=29,
  serialized_end=211,
)

_HEADLINECOLLECTION_HEADLINE.containing_type = _HEADLINECOLLECTION
_HEADLINECOLLECTION.fields_by_name['HL'].message_type = _HEADLINECOLLECTION_HEADLINE
DESCRIPTOR.message_types_by_name['HeadlineCollection'] = _HEADLINECOLLECTION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

HeadlineCollection = _reflection.GeneratedProtocolMessageType('HeadlineCollection', (_message.Message,), {

  'Headline' : _reflection.GeneratedProtocolMessageType('Headline', (_message.Message,), {
    'DESCRIPTOR' : _HEADLINECOLLECTION_HEADLINE,
    '__module__' : 'Headline_pb2'
    # @@protoc_insertion_point(class_scope:Headline.HeadlineCollection.Headline)
    })
  ,
  'DESCRIPTOR' : _HEADLINECOLLECTION,
  '__module__' : 'Headline_pb2'
  # @@protoc_insertion_point(class_scope:Headline.HeadlineCollection)
  })
_sym_db.RegisterMessage(HeadlineCollection)
_sym_db.RegisterMessage(HeadlineCollection.Headline)


# @@protoc_insertion_point(module_scope)
