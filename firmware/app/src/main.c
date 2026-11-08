/* P13 device firmware. Step 1: confirm the toolchain builds and runs our own app.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/version.h>
#ifdef CONFIG_ARCH_POSIX
#include <nsi_main.h>
#endif

int main(void)
{
	printk("P13 FIRMWARE OK (Zephyr %s, uptime %lld ms)\n", KERNEL_VERSION_STRING,
	       (long long)k_uptime_get());
#ifdef CONFIG_ARCH_POSIX
	nsi_exit(0); /* end the native_sim process instead of idling until the timeout */
#endif
	return 0;
}
