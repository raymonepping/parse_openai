# Vault PKI rotation experiment

- Created (UTC): 2024-02-27T02:13:20Z
- Updated (UTC): 2024-02-27T02:21:40Z

## User (2024-02-27T02:13:21Z)

How does Vault PKI certificate rotation work?

## Assistant (2024-02-27T02:13:25Z)

Vault PKI issues certificates with a TTL. When the certificate approaches expiration, a new one can be issued automatically via Vault Agent or through API calls.

## User (2024-02-27T02:13:30Z)

Can you show a CLI example?
