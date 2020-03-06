import re
import glob
import os
import string

_PART_SUFFIX_BASE = '.part'
_PART_SUFFIX_REGEX = _PART_SUFFIX_BASE + r'(\d+)$'


def make_part_suffix(i):
    """Return str that is suffix for index `i`."""
    assert i >= 0, i
    return _PART_SUFFIX_BASE + str(i)


def contains_part_suffix(p):
    """Return true iff path `p` contains a part suffix."""
    return re.search(_PART_SUFFIX_REGEX, p) is not None


def index_of_part_suffix(p):
    """Return index of suffix (final number in path)."""
    return int(re.search(_PART_SUFFIX_REGEX, p).group(1))


def iter_part_suffixes(pin):
    """Return list of all paths that are the same base."""
    assert os.path.isfile(pin)
    assert contains_part_suffix(pin)
    base = pin.rstrip(string.digits)
    matches = glob.glob(base + '*')
    # Filter things of the form base* where * is not digits
    matches = [m for m in matches if contains_part_suffix(m)]
    matches = sorted(matches, key=index_of_part_suffix)
    return matches


def test_part_suffix():
    assert make_part_suffix(1) == '.part1'
    assert make_part_suffix(10) == '.part10'
    assert contains_part_suffix('bla/bli/blupp.part1')
    assert index_of_part_suffix('bla/bli/blupp.part1') == 1
    assert contains_part_suffix('bla/bli/blupp.part3')
    assert not contains_part_suffix('bla/bli/blupp.part3/more')


def test_iter(tmpdir):
    prefix = 'some.file'
    for i in range(16):
        tmpdir.join('some.file' + str(make_part_suffix(i))).write('test')

    some_file = str(tmpdir.join(prefix + str(make_part_suffix(3))))
    assert list(map(os.path.basename, iter_part_suffixes(some_file))) == \
           ['some.file.part{}'.format(i) for i in range(16)]
