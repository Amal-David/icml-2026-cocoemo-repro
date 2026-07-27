# Listening study data dictionary

## Public frozen stimulus manifest

| Field | Meaning |
| --- | --- |
| `stimulus_id` | Opaque public stimulus identifier |
| `item_id` | Content-family identifier |
| `track` | `mixed` or `mismatch` |
| `condition` | Stored only in the frozen investigator manifest until study close |
| `audio_path` | License-cleared generated file path |
| `audio_sha256` | SHA-256 of the exact stimulus bytes |
| `duration_s` | Decoded duration |
| `transcript` | Spoken text |
| `target_allocation` | Five-emotion target distribution |
| `alpha` | Steering strength |
| `backbone` | Generator backbone and pinned revision |
| `vector_id` | Steering-vector identifier and hash |
| `reference_voice_id` | Non-identifying consented reference-voice identifier |
| `generator_commit` | Immutable reproduction repository commit |
| `created_at` | UTC generation timestamp |
| `license_basis` | Redistribution basis for this stimulus |

## Private response object

| Field | Meaning |
| --- | --- |
| `study_id_hmac` | HMAC of the one-time invitation code |
| `protocol_version` | Frozen protocol tag |
| `manifest_hash` | SHA-256 of the frozen stimulus manifest |
| `trial_id` | Opaque presented-trial identifier |
| `audio_sha256` | Hash of the rated audio |
| `server_timestamp` | Server-generated UTC timestamp |
| `playback_duration_ms` | Browser-reported playback duration |
| `naturalness_score` | Integer 1-5 ACR |
| `dominant_emotion` | One of five locked labels |
| `emotion_allocation` | Five integer allocations summing to 100 |
| `attention_results` | Quality-item responses and pass/fail rules |

Plain invitation codes, identities, IP addresses, raw user agents, microphone
data, and free text are prohibited fields.
