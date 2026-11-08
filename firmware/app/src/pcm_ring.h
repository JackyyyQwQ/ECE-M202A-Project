/* Bounded PCM history addressed by absolute sample index (no Zephyr calls).
 *
 * Sample index 0 is the first sample written after boot; indices never wrap
 * (64-bit). The ring keeps only the newest `capacity` samples, so a request
 * [start, start + count) is available only if start >= oldest and
 * start + count <= written. Requests are all-or-nothing.
 *
 * Not thread-safe: the caller serializes writes and reads (one loop for now).
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PCM_RING_H
#define PCM_RING_H

#include <stdbool.h>
#include <stdint.h>

enum pcm_status {
	PCM_OK = 0,
	PCM_OVERWRITTEN = 1,      /* part of the interval was already replaced */
	PCM_NOT_YET_RECORDED = 2, /* part of the interval has not been written */
	PCM_INVALID = 3,          /* count is 0 or larger than the ring */
};

struct pcm_ring {
	int16_t *data;     /* caller-owned storage of `capacity` samples */
	uint32_t capacity;
	uint64_t written;  /* total samples written since boot = next sample index */
};

void pcm_ring_init(struct pcm_ring *ring, int16_t *storage, uint32_t capacity);
void pcm_ring_write(struct pcm_ring *ring, const int16_t *samples, uint32_t count);
/* Index of the oldest sample still held. */
uint64_t pcm_ring_oldest(const struct pcm_ring *ring);
/* Availability of [start, start + count) right now. */
enum pcm_status pcm_ring_check(const struct pcm_ring *ring, uint64_t start, uint32_t count);
/* Copy [start, start + count) into `out` if available; `out` is untouched otherwise. */
enum pcm_status pcm_ring_read(const struct pcm_ring *ring, uint64_t start, uint32_t count,
			      int16_t *out);

/* One audio request being sent in chunks. Only one is active at a time. Each
 * chunk is read from the ring when it is sent, so a slow link can lose the rest
 * of a request to newer audio; the transfer then ends with PCM_OVERWRITTEN. */
struct pcm_transfer {
	bool active;
	uint16_t request_id;
	uint64_t start;
	uint32_t count;
	uint32_t sent;     /* samples already sent */
};

/* Accept a request if the whole interval is available now; otherwise report why. */
enum pcm_status pcm_transfer_begin(struct pcm_transfer *transfer, const struct pcm_ring *ring,
				   uint16_t request_id, uint64_t start, uint32_t count);
/* Copy the next chunk (at most max_samples) into `out`, giving its offset within the
 * request and its length. Returns PCM_OK, or PCM_OVERWRITTEN if that chunk is gone;
 * the transfer becomes inactive after its last chunk or on failure. */
enum pcm_status pcm_transfer_next(struct pcm_transfer *transfer, const struct pcm_ring *ring,
				  uint32_t max_samples, int16_t *out, uint32_t *offset,
				  uint32_t *length);

#endif /* PCM_RING_H */
