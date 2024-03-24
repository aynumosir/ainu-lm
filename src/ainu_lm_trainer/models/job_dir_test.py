from .job_dir import JobDir


def test_constructor() -> None:
    job_dir = JobDir("gs://bucket/foo/bar")
    assert job_dir.blob.bucket.name == "bucket"
    assert job_dir.blob.name == "foo/bar"


def test_resolver() -> None:
    job_dir = JobDir("gs://bucket/foo/bar")
    resolved_job_dir = job_dir.resolve("baz.txt")
    assert resolved_job_dir.blob.bucket.name == "bucket"
    assert resolved_job_dir.blob.name == "foo/bar/baz.txt"


def test_converting_to_str() -> None:
    job_dir = JobDir("gs://bucket/foo/bar")
    assert str(job_dir) == "gs://bucket/foo/bar"
