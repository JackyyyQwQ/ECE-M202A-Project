/* P13 device firmware. Step 2: PCM ring buffer self-test on a synthetic signal.
 *
 * The "recording" is a known sequence: the sample at index i has value
 * (int16_t)(i mod 65536), so any read can be checked exactly.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/version.h>
#ifdef CONFIG_ARCH_POSIX
#include <nsi_main.h>
#endif

#include "pcm_ring.h"

#define SAMPLE_RATE 16000
#define RING_SECONDS 2
#define RING_SAMPLES (SAMPLE_RATE * RING_SECONDS) /* 32,000 samples = 64 KB */
#define FRAME_SAMPLES 1024                        /* one 64 ms model frame */
#define CHUNK_SAMPLES 100                         /* samples per reply chunk */

static int16_t ring_storage[RING_SAMPLES];
static int16_t frame[FRAME_SAMPLES];
static int16_t readback[RING_SAMPLES];
static int failures;

static int16_t expected(uint64_t index)
{
	return (int16_t)(uint16_t)(index & 0xFFFF);
}

static const char *status_name(enum pcm_status status)
{
	static const char *const names[] = {"OK", "OVERWRITTEN", "NOT_YET_RECORDED", "INVALID"};

	return names[status];
}

/* Write `count` samples of the synthetic signal, one frame at a time. */
static void record(struct pcm_ring *ring, uint64_t count)
{
	while (count > 0) {
		uint32_t n = count < FRAME_SAMPLES ? (uint32_t)count : FRAME_SAMPLES;

		for (uint32_t k = 0; k < n; k++) {
			frame[k] = expected(ring->written + k);
		}
		pcm_ring_write(ring, frame, n);
		count -= n;
	}
}

/* Read [start, start + count), compare the status and, if OK, every sample. */
static void check_read(const char *name, const struct pcm_ring *ring, uint64_t start,
		       uint32_t count, enum pcm_status want)
{
	enum pcm_status got = pcm_ring_read(ring, start, count, readback);
	uint32_t bad = 0;

	if (got == PCM_OK) {
		for (uint32_t k = 0; k < count; k++) {
			bad += readback[k] != expected(start + k);
		}
	}
	bool pass = got == want && bad == 0;

	failures += !pass;
	printk("%s %-28s start=%llu count=%u -> %s (want %s), wrong samples=%u\n",
	       pass ? "PASS" : "FAIL", name, (unsigned long long)start, count,
	       status_name(got), status_name(want), bad);
}

/* Request `seconds_of_audio`, then each 64 ms round record one frame and send at
 * most `chunks_per_round` chunks. Returns the final status; checks every sample. */
static enum pcm_status run_transfer(struct pcm_ring *ring, uint32_t chunks_per_round,
				   uint32_t *received)
{
	struct pcm_transfer transfer;
	uint64_t start = ring->written - SAMPLE_RATE; /* the latest 1 s of audio */
	enum pcm_status status = pcm_transfer_begin(&transfer, ring, 1, start, SAMPLE_RATE);
	uint32_t bad = 0;

	*received = 0;
	while (status == PCM_OK && transfer.active) {
		record(ring, FRAME_SAMPLES);
		for (uint32_t c = 0; c < chunks_per_round && transfer.active; c++) {
			uint32_t offset, length;

			status = pcm_transfer_next(&transfer, ring, CHUNK_SAMPLES, readback, &offset,
						   &length);
			for (uint32_t k = 0; k < length; k++) {
				bad += readback[k] != expected(start + offset + k);
			}
			*received += length;
		}
	}
	return bad == 0 ? status : PCM_INVALID;
}

int main(void)
{
	struct pcm_ring ring;

	printk("P13 RING TEST (Zephyr %s): capacity %u samples (%d s), chunk %d samples\n",
	       KERNEL_VERSION_STRING, RING_SAMPLES, RING_SECONDS, CHUNK_SAMPLES);
	pcm_ring_init(&ring, ring_storage, RING_SAMPLES);

	/* 1. Partly filled, nothing overwritten yet. */
	record(&ring, 24000);
	check_read("partly filled", &ring, 1000, 8000, PCM_OK);
	check_read("not yet recorded", &ring, 20000, 8000, PCM_NOT_YET_RECORDED);

	/* 2. After several wraps: 100,000 samples written, oldest is 68,000. */
	record(&ring, 76000);
	uint64_t oldest = pcm_ring_oldest(&ring);
	uint64_t newest = ring.written; /* first index not yet recorded */
	uint64_t wrap = (newest / RING_SAMPLES) * RING_SAMPLES; /* storage restarts here */

	printk("INFO written=%llu oldest=%llu\n", (unsigned long long)newest,
	       (unsigned long long)oldest);
	check_read("across the wrap point", &ring, wrap - 500, 1000, PCM_OK);
	check_read("oldest sample", &ring, oldest, 100, PCM_OK);
	check_read("one sample too old", &ring, oldest - 1, 100, PCM_OVERWRITTEN);
	check_read("newest sample", &ring, newest - 100, 100, PCM_OK);
	check_read("one sample in the future", &ring, newest - 99, 100, PCM_NOT_YET_RECORDED);
	check_read("whole ring", &ring, oldest, RING_SAMPLES, PCM_OK);
	check_read("empty request", &ring, oldest, 0, PCM_INVALID);
	check_read("longer than the ring", &ring, oldest, RING_SAMPLES + 1, PCM_INVALID);

	/* 3. Chunked transfer of 1 s (160 chunks) while recording continues. */
	struct {
		uint32_t chunks_per_round;
		enum pcm_status want;
	} links[] = {
		{40, PCM_OK},          /* fast link: done in 4 rounds */
		{10, PCM_OK},          /* 16 rounds ~ 1 s, still inside the 2 s ring */
		{2, PCM_OVERWRITTEN},  /* 80 rounds ~ 5 s: the tail is overwritten */
	};
	for (size_t i = 0; i < ARRAY_SIZE(links); i++) {
		uint32_t received;
		enum pcm_status got = run_transfer(&ring, links[i].chunks_per_round, &received);
		bool pass = got == links[i].want;

		failures += !pass;
		printk("%s transfer, %2u chunks/round   -> %s (want %s), received %u of %d samples\n",
		       pass ? "PASS" : "FAIL", links[i].chunks_per_round, status_name(got),
		       status_name(links[i].want), received, SAMPLE_RATE);
	}

	printk("P13 RING TEST %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
#ifdef CONFIG_ARCH_POSIX
	nsi_exit(failures ? 1 : 0);
#endif
	return failures ? 1 : 0;
}
