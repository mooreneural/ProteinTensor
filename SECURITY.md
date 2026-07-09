# Security Policy

## Supported versions

ProteinTensor is pre-1.0 and under active development. Security fixes are applied
to the latest released version on PyPI.

| Version | Supported |
|---------|-----------|
| 0.4.x   | Yes       |
| < 0.4   | No        |

Please upgrade to the latest release (`pip install --upgrade proteintensor`)
before reporting an issue.

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, report them privately by one of:

- Email: **claytonwaynemoore@gmail.com** with the subject
  "ProteinTensor Security".
- GitHub's private [security advisory](https://github.com/mooreneural/ProteinTensor/security/advisories/new)
  form ("Report a vulnerability").

Please include:

- A description of the vulnerability and its impact.
- Steps to reproduce, or a proof-of-concept.
- The ProteinTensor version and environment (OS, Python version).

## What to expect

- Acknowledgement of your report within 5 business days.
- An assessment and, where applicable, a fix in a patch release.
- Credit for the discovery if you would like it, once a fix is released.

## Scope

ProteinTensor reads and writes `.ptt` (Zarr) files and can stream from object
storage. Relevant concerns include, but are not limited to: unsafe handling of
untrusted `.ptt` files or crafted structure/MSA inputs, and issues in the
optional cloud (`fsspec`/`s3fs`/`gcsfs`) and RDKit code paths. Vulnerabilities in
third-party dependencies should be reported to those projects, though we welcome
a heads-up if they affect ProteinTensor users.
