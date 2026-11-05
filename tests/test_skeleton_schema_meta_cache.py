from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import sys

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from unchain.skeleton import kernel_output as output
from unchain.skeleton.errors import SkeletonValidationError


def schema(**fields):
    return {'type': 'object', 'properties': {'answer': {'type': 'string'}}, **fields}


@pytest.fixture(autouse=True)
def clear_cache():
    cache = getattr(output, '_SCHEMA_META_CACHE', None)
    if cache is not None:
        cache.clear()
    yield
    if cache is not None:
        cache.clear()


def native_calls(operation):
    code = Draft202012Validator.check_schema.__func__.__code__
    count = 0
    previous = sys.getprofile()
    def observe(frame, event, arg):
        nonlocal count
        if event == 'call' and frame.f_code is code:
            count += 1
    sys.setprofile(observe)
    try:
        operation()
    finally:
        sys.setprofile(previous)
    return count


def test_native_meta_checker_runs_once_for_equal_detached_schemas():
    results = []
    def twice():
        results.extend(output.validate_output_schema(schema()) for _ in range(2))
    assert native_calls(twice) == 1
    assert results[0] == results[1]
    assert results[0] is not results[1]
    assert results[0]['properties'] is not results[1]['properties']


def test_warm_cache_does_not_skip_tree_walk_refs_or_detachment(monkeypatch):
    source = schema()
    admitted = output.validate_output_schema(source)
    admitted['properties']['answer']['type'] = 'integer'
    assert output.validate_output_schema(source)['properties']['answer']['type'] == 'string'
    source['properties']['answer']['pattern'] = 'not supported'
    with pytest.raises(SkeletonValidationError, match='unsupported'):
        output.validate_output_schema(source)
    source = schema()
    output.validate_output_schema(source)
    # A warm meta cache cannot replace per-call admission.
    def reject(*args, **kwargs):
        raise SkeletonValidationError('test_walker', ('schema',), 'walker invoked')
    monkeypatch.setattr(output._SchemaWalker, 'resolve_references', reject)
    with pytest.raises(SkeletonValidationError) as error:
        output.validate_output_schema(source)
    assert error.value.code == 'test_walker'


def test_mutated_source_changes_exact_bytes_and_checks_again():
    source = schema()
    def mutate():
        output.validate_output_schema(source)
        source['description'] = 'changed'
        output.validate_output_schema(source)
    assert native_calls(mutate) == 2
    source['properties']['message'] = {'type': 'string'}
    with pytest.raises(SkeletonValidationError) as error:
        output.validate_output_schema(source)
    assert error.value.code == 'reserved_name'


def test_order_number_unicode_surrogate_and_reference_bytes_remain_distinct():
    values = [schema(description='中\ud800\n'), dict(reversed(list(schema(description='中\ud800\n').items()))),
              schema(minimum=1), schema(minimum=1.0),
              schema(**{'$defs': {'x': {'type': 'string'}}, 'anyOf': [{'$ref': '#/$defs/x'}]})]
    assert native_calls(lambda: [output.validate_output_schema(value) for value in values]) == len(values)
    assert native_calls(lambda: [output.validate_output_schema(deepcopy(value)) for value in values]) == 0


@pytest.mark.parametrize('exception', [SchemaError('invalid'), TypeError('invalid'), ValueError('invalid'), RecursionError('invalid')])
def test_replaced_checker_is_current_uncached_and_keeps_error_translation(monkeypatch, exception):
    source = schema()
    output.validate_output_schema(source)
    calls = []
    def checker(cls, value):
        calls.append(value)
        raise exception
    with monkeypatch.context() as patch:
        patch.setattr(Draft202012Validator, 'check_schema', classmethod(checker))
        for _ in range(2):
            with pytest.raises(SkeletonValidationError) as error:
                output.validate_output_schema(source)
            assert error.value.code == 'invalid_schema'
            assert error.value.path == ('schema',)
            assert error.value.__cause__ is exception
    assert len(calls) == 2
    assert native_calls(lambda: output.validate_output_schema(source)) == 0
    assert output.validate_output_schema(source) == source


def test_replaced_class_bypasses_even_if_checker_function_is_same(monkeypatch):
    source = schema()
    output.validate_output_schema(source)
    class Other(Draft202012Validator):
        pass
    monkeypatch.setattr(output, 'Draft202012Validator', Other)
    assert native_calls(lambda: output.validate_output_schema(source)) == 1


def test_meta_failures_are_not_cached(monkeypatch):
    # Keep the real checker function and exercise its native SchemaError path.
    meta = deepcopy(Draft202012Validator.META_SCHEMA)
    meta['required'] = ['missing_test_annotation']
    monkeypatch.setattr(Draft202012Validator, 'META_SCHEMA', meta)
    source = schema()
    def twice():
        for _ in range(2):
            with pytest.raises(SkeletonValidationError):
                output.validate_output_schema(source)
    assert native_calls(twice) == 2


def test_warm_cache_bypasses_replaced_native_meta_schema(monkeypatch):
    source = schema()
    output.validate_output_schema(source)
    meta = deepcopy(Draft202012Validator.META_SCHEMA)
    meta['required'] = ['missing_test_annotation']
    monkeypatch.setattr(Draft202012Validator, 'META_SCHEMA', meta)
    def twice():
        for _ in range(2):
            with pytest.raises(SkeletonValidationError) as error:
                output.validate_output_schema(source)
            assert isinstance(error.value.__cause__, SchemaError)
    assert native_calls(twice) == 2


def test_warm_cache_bypasses_replaced_native_format_checker(monkeypatch):
    from jsonschema import FormatChecker
    source = schema()
    output.validate_output_schema(source)
    monkeypatch.setattr(Draft202012Validator, 'FORMAT_CHECKER', FormatChecker())
    assert native_calls(lambda: output.validate_output_schema(source)) == 1


def test_warm_cache_bypasses_in_place_native_meta_schema_mutation(monkeypatch):
    source = schema()
    output.validate_output_schema(source)
    monkeypatch.setitem(Draft202012Validator.META_SCHEMA, 'required', ['missing_test_annotation'])
    def twice():
        for _ in range(2):
            with pytest.raises(SkeletonValidationError) as error:
                output.validate_output_schema(source)
            assert error.value.code == 'invalid_schema'
            assert error.value.path == ('schema',)
            assert isinstance(error.value.__cause__, SchemaError)
    assert native_calls(twice) == 2


def test_warm_cache_bypasses_in_place_native_format_registration(monkeypatch):
    source = schema(**{'$schema': 'https://json-schema.org/draft/2020-12/schema'})
    output.validate_output_schema(source)
    monkeypatch.setitem(Draft202012Validator.FORMAT_CHECKER.checkers, 'uri', (lambda value: False, ()))
    def twice():
        for _ in range(2):
            with pytest.raises(SkeletonValidationError) as error:
                output.validate_output_schema(source)
            assert error.value.code == 'invalid_schema'
            assert error.value.path == ('schema',)
            assert isinstance(error.value.__cause__, SchemaError)
    assert native_calls(twice) == 2


def test_sixteen_kib_boundary_and_sixty_four_entry_eviction():
    empty = schema(description='')
    overhead = len(output._enforce_schema_size(empty))
    edge = schema(description='x' * (16 * 1024 - overhead))
    above = schema(description='x' * (16 * 1024 - overhead + 1))
    assert native_calls(lambda: [output.validate_output_schema(edge) for _ in range(2)]) == 1
    assert native_calls(lambda: [output.validate_output_schema(above) for _ in range(2)]) == 2
    output._SCHEMA_META_CACHE.clear()
    for index in range(65):
        output.validate_output_schema(schema(description=str(index)))
    assert len(output._SCHEMA_META_CACHE) == 64
    assert all(type(key) is bytes and value is None for key, value in output._SCHEMA_META_CACHE.items())
    assert native_calls(lambda: output.validate_output_schema(schema(description='0'))) == 1


def test_concurrent_calls_are_bounded_and_detached():
    sources = [schema(description=str(index % 70)) for index in range(140)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(output.validate_output_schema, sources))
    assert results == sources
    assert len(output._SCHEMA_META_CACHE) <= 64
    results[0]['properties']['answer']['type'] = 'integer'
    assert results[70]['properties']['answer']['type'] == 'string'
    assert sources[0]['properties']['answer']['type'] == 'string'
