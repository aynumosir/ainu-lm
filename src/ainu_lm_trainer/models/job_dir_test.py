from .job_dir import JobDir


def test_constructor() -> None:
    job_dir = JobDir("gs://bucket/foo/bar")
    blob = job_dir.to_blob(None)
    assert blob.bucket.name == "bucket"
    assert blob.name == "foo/bar"


def test_resolver() -> None:
    job_dir = JobDir("gs://bucket/foo/bar")
    resolved_job_dir = job_dir.resolve("baz.txt")
    blob = resolved_job_dir.to_blob(None)
    assert blob.bucket.name == "bucket"
    assert blob.name == "foo/bar/baz.txt"


def test_converting_to_str() -> None:
    job_dir = JobDir("gs://bucket/foo/bar")
    assert str(job_dir) == "gs://bucket/foo/bar"


def test_trailing_slash() -> None:
    job_dir = JobDir("gs://bucket/foo/bar/")
    assert str(job_dir) == "gs://bucket/foo/bar"
    resolved_job_dir = job_dir.resolve("baz.txt")
    blob = resolved_job_dir.to_blob(None)
    assert blob.bucket.name == "bucket"
    assert blob.name == "foo/bar/baz.txt"
