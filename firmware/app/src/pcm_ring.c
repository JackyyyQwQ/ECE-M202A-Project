/* Bounded PCM history; see pcm_ring.h. SPDX-License-Identifier: Apache-2.0 */

#include "pcm_ring.h"

#include <string.h>

void pcm_ring_init(struct pcm_ring *ring, int16_t *storage, uint32_t capacity)
{
	ring->data = storage;
	ring->capacity = capacity;
	ring->written = 0;
}

void pcm_ring_write(struct pcm_ring *ring, const int16_t *samples, uint32_t count)
{
	/* Copy in at most two pieces: up to the end of storage, then from its start. */
	while (count > 0) {
		uint32_t position = (uint32_t)(ring->written % ring->capacity);
		uint32_t piece = ring->capacity - position;

		if (piece > count) {
			piece = count;
		}
		memcpy(&ring->data[position], samples, piece * sizeof(int16_t));
		ring->written += piece;
		samples += piece;
		count -= piece;
	}
}

uint64_t pcm_ring_oldest(const struct pcm_ring *ring)
{
	return ring->written > ring->capacity ? ring->written - ring->capacity : 0;
}

enum pcm_status pcm_ring_check(const struct pcm_ring *ring, uint64_t start, uint32_t count)
{
	if (count == 0 || count > ring->capacity) {
		return PCM_INVALID;
	}
	if (start < pcm_ring_oldest(ring)) {
		return PCM_OVERWRITTEN;
	}
	if (start + count > ring->written) {
		return PCM_NOT_YET_RECORDED;
	}
	return PCM_OK;
}

enum pcm_status pcm_ring_read(const struct pcm_ring *ring, uint64_t start, uint32_t count,
			      int16_t *out)
{
	enum pcm_status status = pcm_ring_check(ring, start, count);

	if (status != PCM_OK) {
		return status;
	}
	uint32_t position = (uint32_t)(start % ring->capacity);
	uint32_t first = ring->capacity - position;

	if (first > count) {
		first = count;
	}
	memcpy(out, &ring->data[position], first * sizeof(int16_t));
	memcpy(out + first, ring->data, (count - first) * sizeof(int16_t));
	return PCM_OK;
}

enum pcm_status pcm_transfer_begin(struct pcm_transfer *transfer, const struct pcm_ring *ring,
				   uint16_t request_id, uint64_t start, uint32_t count)
{
	enum pcm_status status = pcm_ring_check(ring, start, count);

	transfer->active = (status == PCM_OK);
	transfer->request_id = request_id;
	transfer->start = start;
	transfer->count = count;
	transfer->sent = 0;
	return status;
}

enum pcm_status pcm_transfer_next(struct pcm_transfer *transfer, const struct pcm_ring *ring,
				  uint32_t max_samples, int16_t *out, uint32_t *offset,
				  uint32_t *length)
{
	uint32_t remaining = transfer->count - transfer->sent;
	uint32_t n = remaining < max_samples ? remaining : max_samples;

	*offset = transfer->sent;
	*length = 0;
	if (!transfer->active || n == 0) {
		transfer->active = false;
		return PCM_INVALID;
	}
	enum pcm_status status = pcm_ring_read(ring, transfer->start + transfer->sent, n, out);

	if (status != PCM_OK) {
		transfer->active = false; /* the rest of the request cannot be sent */
		return status;
	}
	*length = n;
	transfer->sent += n;
	transfer->active = transfer->sent < transfer->count;
	return PCM_OK;
}
