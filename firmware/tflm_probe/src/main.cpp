/* TFLM size probe for the distilled Nano encoder (one per-frame window).
 * Prints the model size, the tensor arena actually used, the tensor shapes and
 * whether one inference on an all-zero input succeeds. native_sim cycle counts
 * are not Nano timings. */
#include <string.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>
#ifdef CONFIG_ARCH_POSIX
#include <nsi_main.h>
#endif

#include P13_MODEL_HEADER

alignas(16) static uint8_t tensor_arena[P13_ARENA_KB * 1024];

static void print_shape(const char *label, const TfLiteTensor *t)
{
	printk("  %s:", label);
	for (int i = 0; i < t->dims->size; i++) {
		printk(" %d", t->dims->data[i]);
	}
	printk(" (%u bytes, type %d)\n", (unsigned)t->bytes, t->type);
}

static int run(void)
{
	const tflite::Model *model = tflite::GetModel(g_model);
	if (model->version() != TFLITE_SCHEMA_VERSION) {
		printk("P13 TFLM FAIL: schema %u != %d\n", (unsigned)model->version(), TFLITE_SCHEMA_VERSION);
		return 1;
	}
	static tflite::MicroMutableOpResolver<8> resolver;
	resolver.AddPad();
	resolver.AddConv2D();
	resolver.AddDepthwiseConv2D();
	resolver.AddAveragePool2D();
	resolver.AddMean();
	resolver.AddReshape();
	resolver.AddQuantize();
	resolver.AddDequantize();

	static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, sizeof(tensor_arena));
	if (interpreter.AllocateTensors() != kTfLiteOk) {
		printk("P13 TFLM FAIL: AllocateTensors (arena %d KB too small or op missing)\n", P13_ARENA_KB);
		return 1;
	}
	TfLiteTensor *in = interpreter.input(0);
	memset(in->data.int8, in->params.zero_point, in->bytes);  /* all-zero mel window */
	uint32_t start = k_cycle_get_32();
	TfLiteStatus status = interpreter.Invoke();
	uint32_t cycles = k_cycle_get_32() - start;

	printk("P13 TFLM model=%s\n", P13_MODEL_NAME);
	printk("  model_bytes=%u arena_used=%u arena_reserved=%u\n", g_model_len,
	       (unsigned)interpreter.arena_used_bytes(), (unsigned)sizeof(tensor_arena));
	print_shape("input", in);
	print_shape("output", interpreter.output(0));
	printk("  invoke=%s cycles=%u (native_sim: not a Nano timing)\n",
	       status == kTfLiteOk ? "ok" : "FAILED", cycles);
	printk("P13 TFLM %s\n", status == kTfLiteOk ? "PASS" : "FAIL");
	return status == kTfLiteOk ? 0 : 1;
}

int main(void)
{
	int rc = run();
#ifdef CONFIG_ARCH_POSIX
	nsi_exit(rc);
#endif
	return rc;
}
