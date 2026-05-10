# Contributing to Rustvani

Thank you for your interest in contributing to Rustvani — a Rust-based voice agent pipeline framework. Whether you're adding a new STT/TTS integration, improving documentation, fixing bugs, or building examples, your contributions are welcome.

## Table of Contents

- [Getting Started](#getting-started)
- [Contribution Priorities](#contribution-priorities)
- [Adding New Integrations](#adding-new-integrations)
- [Project Structure](#project-structure)
- [Pull Request Workflow](#pull-request-workflow)
- [Code Style](#code-style)
- [Documentation](#documentation)
- [Examples](#examples)
- [Code of Conduct](#code-of-conduct)

---

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/rustvani.git
   cd rustvani
   ```
3. **Build** the project:
   ```bash
   cargo build
   ```
4. **Run tests:**
   ```bash
   cargo test
   ```

Make sure you have a recent stable Rust toolchain installed. The project uses Tokio for async and axum for WebSocket transport.

---

## Contribution Priorities

These are the areas where contributions will have the most impact, roughly in order of priority:

### 1. New STT / TTS / LLM Integrations

Rustvani currently supports Sarvam and OpenAI. We want to expand this significantly. Every major voice AI provider should have a first-class integration. Some specific targets:

- **Smallest AI** — TTS and STT
- **Deepgram** — STT and TTS
- **ElevenLabs** — TTS
- **Azure Cognitive Services** — STT and TTS
- **Google Cloud Speech** — STT and TTS
- **AWS Transcribe / Polly** — STT and TTS
- **Whisper (local)** — STT
- **Piper (local)** — TTS
- **Kokoro (local)** — TTS

If a provider isn't listed here, it's still welcome. Open an issue first to discuss.

### 2. Custom Silero VAD Inference

The current VAD implementation uses Silero via the `ort` ONNX runtime crate. A target goal is implementing custom Silero VAD inference directly in Rust — reducing the ONNX dependency and giving us finer control over the detection pipeline. This is a substantial but high-impact contribution.

### 3. Documentation Overhaul

Current documentation is AI-generated and needs significant improvement:

- **Accuracy passes** — verify every doc against actual code behavior
- **Architecture docs** — explain the pipeline model, frame flow, and Dhara conversation state machine
- **Integration guides** — step-by-step guides for wiring up each STT/TTS/LLM provider
- **API reference** — ensure all public types, traits, and functions have clear, accurate rustdoc comments
- **Inline code comments** — add context and rationale where the code isn't self-explanatory

If you spot a doc that's wrong or misleading, fixing it is a valuable contribution even if it's a one-line change.

### 4. Examples

Good examples are the fastest way for new users to understand rustvani. We want:

- **Minimal echo bot** — simplest possible pipeline: STT → echo text back → TTS
- **Multi-provider swap** — same bot, different STT/TTS backends, showing how to switch
- **Function calling demo** — a bot that calls external APIs based on conversation (like the pizza ordering demo, but generalized)
- **Interruption handling demo** — showcasing barge-in and graceful interruption
- **Multi-language bot** — a bot that detects language and switches TTS voice accordingly
- **WebSocket client example** — a minimal browser client connecting to a rustvani server

Examples should be self-contained, well-commented, and runnable with minimal setup.

---

## Adding New Integrations

**This is the most common type of contribution.** Before you start, study the existing integrations as reference implementations:

- **Sarvam** — `src/sarvam/` (STT, TTS, and LLM)
- **OpenAI** — `src/openai/` (LLM)

These are the canonical examples. New integrations should follow their patterns closely.

### General approach

1. **Create a module** under `src/<provider>/` with separate files for each capability (e.g., `stt.rs`, `tts.rs`).
2. **Implement the relevant traits** — your STT processor must implement the STT pipeline trait, your TTS processor must implement the TTS pipeline trait. Look at how Sarvam does it.
3. **Handle streaming correctly** — most providers offer streaming APIs. Use them. Rustvani is a real-time pipeline; buffering entire responses defeats the purpose.
4. **Error handling** — use proper Rust error types. No `.unwrap()` in library code. Propagate errors up the pipeline gracefully.
5. **Configuration** — provider config (API keys, endpoints, model names, voice IDs) should be passed via a config struct, not hardcoded.
6. **Feature flags** — gate your integration behind a Cargo feature flag so it doesn't bloat builds for users who don't need it:
   ```toml
   [features]
   smallest-ai = ["dep:reqwest"]  # example
   ```
7. **Tests** — at minimum, unit tests for serialization/deserialization of API types. Integration tests that hit live APIs should be behind an `#[ignore]` attribute.
8. **Documentation** — rustdoc on all public items. A short README or doc comment at the module level explaining what the integration covers and any provider-specific quirks.

### Checklist for new integrations

- [ ] Module created under `src/<provider>/`
- [ ] Implements the correct pipeline traits
- [ ] Streams responses where the provider API supports it
- [ ] Config struct for all provider-specific settings
- [ ] Gated behind a Cargo feature flag
- [ ] No `.unwrap()` in library code
- [ ] Unit tests for API types
- [ ] Rustdoc on all public items
- [ ] Example added or existing example updated to demonstrate usage

---

## Project Structure

```
rustvani/
├── src/
│   ├── pipeline/       # Core pipeline traits and frame types
│   ├── transport/      # WebSocket transport (axum-based)
│   ├── vad/            # Voice Activity Detection (Silero/ONNX)
│   ├── dhara/          # Conversation flow state machine
│   ├── sarvam/         # Sarvam AI integration (reference impl)
│   ├── openai/         # OpenAI integration (reference impl)
│   └── ...
├── examples/           # Runnable demo bots
├── docs/               # Documentation
└── Cargo.toml
```

When adding a new integration, your code should live under `src/<provider>/`. Don't modify core pipeline code unless your integration genuinely requires it — and if it does, discuss it in the issue first.

---

## Pull Request Workflow

1. **Open an issue first** for non-trivial changes. Describe what you want to do and why. This avoids wasted effort if the approach needs adjustment.
2. **Fork** the repository.
3. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feat/smallest-ai-tts
   ```
4. **Make your changes.** Commit in logical chunks with clear messages.
5. **Run checks locally:**
   ```bash
   cargo fmt --check
   cargo clippy -- -D warnings
   cargo test
   ```
6. **Push** your branch and **open a Pull Request** against `main`.
7. **Describe your PR** — what it does, why, and any decisions you made. Link the related issue.
8. **Respond to review feedback.** PRs typically need at least one approval before merge.

### Branch naming conventions

- `feat/<description>` — new features or integrations
- `fix/<description>` — bug fixes
- `docs/<description>` — documentation changes
- `refactor/<description>` — code restructuring without behavior changes
- `example/<description>` — new or updated examples

---

## Code Style

- **Run `cargo fmt`** before committing. No exceptions.
- **Run `cargo clippy`** and fix all warnings.
- **No `.unwrap()` or `.expect()` in library code.** Use proper error propagation with `Result` and `?`. Panics are acceptable only in examples or tests where the intent is clear.
- **Async by default.** Rustvani is a Tokio-based async pipeline. Blocking calls must be wrapped in `tokio::task::spawn_blocking`.
- **Naming** — follow Rust conventions: `snake_case` for functions and variables, `PascalCase` for types and traits, `SCREAMING_SNAKE_CASE` for constants.
- **Keep modules focused.** One file per concern. If a file is growing past ~400 lines, consider splitting it.
- **Comments** — explain *why*, not *what*. The code should be readable enough that the *what* is obvious.

---

## Documentation

As noted above, current documentation is largely AI-generated and needs human review. When contributing docs:

- **Verify against the code.** If a doc says a function does X, read the function and confirm it actually does X.
- **Be specific.** "This processes audio frames" is not useful. "This receives PCM16 audio frames from the transport, runs them through VAD, and forwards voiced segments to the STT processor" is.
- **Include examples** in rustdoc where it helps:
  ```rust
  /// Creates a new Sarvam TTS processor.
  ///
  /// # Example
  /// ```no_run
  /// let tts = SarvamTts::new(SarvamTtsConfig {
  ///     api_key: "your-key".into(),
  ///     voice: "meera".into(),
  ///     ..Default::default()
  /// });
  /// ```
  ```
- **Don't generate docs with AI and submit them as-is.** If you use AI to draft, review and edit thoroughly. We're trying to move *away* from unreviewed AI-generated docs.

---

## Examples

Examples live in the `examples/` directory. A good example:

- **Runs with minimal setup** — ideally just an API key in an env var
- **Is self-contained** — one file, no external dependencies beyond what's in `Cargo.toml`
- **Has comments** explaining the pipeline setup and what each component does
- **Shows real usage** — not toy code, but something someone could adapt for a real application
- **Has a README section or top-of-file doc comment** explaining what it demonstrates and how to run it

When adding a new integration, adding or updating an example to demonstrate it is strongly encouraged.

---

## Code of Conduct

### Our Commitment

We are committed to providing a welcoming and respectful environment for everyone, regardless of experience level, background, identity, or perspective.

### Expected Behavior

- Be respectful and constructive in discussions, issues, and code reviews.
- Give and receive feedback gracefully. Critique code, not people.
- Assume good intent. If something reads ambiguously, ask for clarification before reacting.
- Help others learn. If someone's contribution needs work, explain what and why — don't just reject.

### Unacceptable Behavior

- Personal attacks, insults, or derogatory language.
- Harassment of any kind.
- Publishing others' private information without consent.
- Dismissing contributions based on the contributor's experience level.

### Enforcement

Project maintainers may remove, edit, or reject comments, commits, code, issues, and other contributions that violate this Code of Conduct. Repeated or severe violations may result in a temporary or permanent ban from the project.

### Scope

This Code of Conduct applies to all project spaces — GitHub issues, pull requests, discussions, and any other communication channels associated with Rustvani.

---

## Questions?

If you're unsure about anything — whether an integration is wanted, how to structure something, or where to start — open an issue and ask. We'd rather help you get started than have you struggle in silence.

Thank you for contributing to Rustvani.
