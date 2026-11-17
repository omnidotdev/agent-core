# Changesets

Add a changeset for any change a consumer of this crate would notice: `bunx changeset`.

Merging to `master` opens (or updates) a "version packages" PR. Merging that PR bumps `package.json` and `Cargo.toml`, updates `CHANGELOG.md`, and tags the release (`vX.Y.Z`) with a GitHub Release. Consumers pin agent-core as a git dependency, so a release is a tag, never a registry publish.
