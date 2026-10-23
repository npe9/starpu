/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2026  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu.h>
#include <core/simgrid.h>
#ifdef STARPU_DEBUG
#include <core/workers.h>
#endif
#include <common/thread.h>
#include <profiling/starpu_tracing.h>
#include <common/timing.h>

#include <errno.h>
#include <limits.h>

#ifdef STARPU_SIMGRID
#ifdef STARPU_HAVE_SIMGRID_MUTEX_H
#include <simgrid/mutex.h>
#include <simgrid/cond.h>
#elif defined(STARPU_HAVE_XBT_SYNCHRO_H)
#include <xbt/synchro.h>
#else
#include <xbt/synchro_core.h>
#endif
#include <smpi/smpi.h>
#if !defined(HAVE_SG_ACTOR_GET_DATA) && !defined(HAVE_SG_ACTOR_DATA) && \
	(defined(HAVE_SMPI_PROCESS_SET_USER_DATA) || defined(smpi_process_get_user_data)) && \
	!(defined(HAVE_MSG_PROCESS_SELF_NAME) || defined(MSG_process_self_name))
#include <simgrid/simix.h>
#endif
#else

#if defined(STARPU_LINUX_SYS) && defined(STARPU_HAVE_XCHG)
#include <linux/futex.h>
#include <sys/syscall.h>
#if defined(STARPU_LITHE) && !defined(STARPU_FXT_LOCK_TRACES) && !defined(_MSC_VER)
/* Lithe port helpers, defined with the rest of the port below. */
static int _starpu_lithe_coop(void);
extern int starpu_lithe_futex_wait_(int *, int) __asm__("futex_wait") __attribute__((weak));
extern int starpu_lithe_futex_wake_some_(int *, int) __asm__("futex_wake_some") __attribute__((weak));
#define _STARPU_SPIN_LITHE 1
#endif

/* Private futexes are not so old, cope with old kernels.  */
#ifdef FUTEX_WAIT_PRIVATE
static int _starpu_futex_wait = FUTEX_WAIT_PRIVATE;
static int _starpu_futex_wake = FUTEX_WAKE_PRIVATE;
#else
static int _starpu_futex_wait = FUTEX_WAIT;
static int _starpu_futex_wake = FUTEX_WAKE;
#endif

#endif
#endif /* !STARPU_SIMGRID */

#ifdef STARPU_SIMGRID

int starpu_pthread_equal(starpu_pthread_t t1, starpu_pthread_t t2)
{
	return t1 == t2;
}

starpu_pthread_t starpu_pthread_self(void)
{
#ifdef HAVE_SG_ACTOR_SELF
	return sg_actor_self();
#else
	return MSG_process_self();
#endif
}

int starpu_pthread_create_on(const char *name, starpu_pthread_t *thread, const starpu_pthread_attr_t *attr STARPU_ATTRIBUTE_UNUSED, void *(*start_routine) (void *), void *arg, starpu_sg_host_t host)
{
	char **_args;
	int ret;
	_STARPU_MALLOC(_args, 3*sizeof(char*));
	ret = asprintf(&_args[0], "%p", start_routine);
	STARPU_ASSERT(ret);
	ret = asprintf(&_args[1], "%p", arg);
	STARPU_ASSERT(ret);
	_args[2] = NULL;
	if (!host)
		host = _starpu_simgrid_get_host_by_name("MAIN");

	void *tsd;
	_STARPU_CALLOC(tsd, MAX_TSD+1, sizeof(void*));

#ifndef HAVE_SG_ACTOR_SET_STACKSIZE
	if (attr && attr->stacksize)
		_starpu_simgrid_set_stack_size(attr->stacksize);
#endif
#ifdef HAVE_SG_ACTOR_INIT
	*thread= sg_actor_init(name, host);
#ifdef HAVE_SG_ACTOR_SET_STACKSIZE
	if (attr && attr->stacksize)
		sg_actor_set_stacksize(*thread, attr->stacksize);
#endif
#ifdef HAVE_SG_ACTOR_SET_DATA
	sg_actor_set_data(*thread, tsd);
#else
	sg_actor_data_set(*thread, tsd);
#endif
	sg_actor_start(*thread, _starpu_simgrid_thread_start, 2, _args);
#else
	*thread = MSG_process_create_with_arguments(name, _starpu_simgrid_thread_start, tsd, host, 2, _args);
#if defined(HAVE_SG_ACTOR_DATA) || defined(HAVE_SG_ACTOR_GET_DATA)
#ifdef HAVE_SG_ACTOR_SET_DATA
	sg_actor_set_data(*thread, tsd);
#else
	sg_actor_data_set(*thread, tsd);
#endif
#endif
#endif
#ifndef HAVE_SG_ACTOR_SET_STACKSIZE
	if (attr && attr->stacksize)
		_starpu_simgrid_set_stack_size(_starpu_default_stack_size);
#endif

#if SIMGRID_VERSION >= 31500 && SIMGRID_VERSION != 31559
#  ifdef HAVE_SG_ACTOR_REF
	sg_actor_ref(*thread);
#  else
	MSG_process_ref(*thread);
#  endif
#endif
	return 0;
}

int starpu_pthread_create(starpu_pthread_t *thread, const starpu_pthread_attr_t *attr, void *(*start_routine) (void *), void *arg)
{
	return starpu_pthread_create_on("", thread, attr, start_routine, arg, NULL);
}

int starpu_pthread_join(starpu_pthread_t thread STARPU_ATTRIBUTE_UNUSED, void **retval STARPU_ATTRIBUTE_UNUSED)
{
#if SIMGRID_VERSION >= 31400
#  ifdef STARPU_HAVE_SIMGRID_ACTOR_H
	sg_actor_join(thread, 1000000);
#  else
	MSG_process_join(thread, 1000000);
#  endif
#if SIMGRID_VERSION >= 31500 && SIMGRID_VERSION != 31559
#  ifdef HAVE_SG_ACTOR_REF
	sg_actor_unref(thread);
#  else
	MSG_process_unref(thread);
#  endif
#endif
#else
	starpu_sleep(1);
#endif
	return 0;
}

int starpu_pthread_detach(starpu_pthread_t thread STARPU_ATTRIBUTE_UNUSED)
{
#if SIMGRID_VERSION >= 31500 && SIMGRID_VERSION != 31559
#  ifdef HAVE_SG_ACTOR_REF
	sg_actor_unref(thread);
#  else
	MSG_process_unref(thread);
#  endif
#endif
	return 0;
}

int starpu_pthread_exit(void *retval STARPU_ATTRIBUTE_UNUSED)
{
#ifdef HAVE_SG_ACTOR_SELF
	sg_actor_kill(sg_actor_self());
#else
	MSG_process_kill(MSG_process_self());
#endif
	STARPU_ABORT_MSG("MSG_process_kill(MSG_process_self()) returned?!");
}

int starpu_pthread_attr_init(starpu_pthread_attr_t *attr)
{
	attr->stacksize = 0;
	return 0;
}

int starpu_pthread_attr_destroy(starpu_pthread_attr_t *attr STARPU_ATTRIBUTE_UNUSED)
{
	return 0;
}

int starpu_pthread_attr_setstacksize(starpu_pthread_attr_t *attr, size_t stacksize)
{
	attr->stacksize = stacksize;
	return 0;
}

int starpu_pthread_attr_setdetachstate(starpu_pthread_attr_t *attr STARPU_ATTRIBUTE_UNUSED, int detachstate STARPU_ATTRIBUTE_UNUSED)
{
	return 0;
}

int starpu_pthread_mutex_init(starpu_pthread_mutex_t *mutex, const starpu_pthread_mutexattr_t *mutexattr STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_HAVE_SIMGRID_MUTEX_H
	*mutex = sg_mutex_init();
#else
	*mutex = xbt_mutex_init();
#endif
	return 0;
}

int starpu_pthread_mutex_destroy(starpu_pthread_mutex_t *mutex)
{
	if (*mutex)
#ifdef STARPU_HAVE_SIMGRID_MUTEX_H
		sg_mutex_destroy(*mutex);
#else
		xbt_mutex_destroy(*mutex);
#endif
	return 0;
}

int starpu_pthread_mutex_lock(starpu_pthread_mutex_t *mutex)
{
	_starpu_trace_locking_mutex();

	/* Note: this is actually safe, because simgrid only preempts within
	 * simgrid functions */
	if (!*mutex)
	{
		/* Here we may get preempted */
#ifdef STARPU_HAVE_SIMGRID_MUTEX_H
		sg_mutex_t new_mutex = sg_mutex_init();
#else
		xbt_mutex_t new_mutex = xbt_mutex_init();
#endif
		if (!*mutex)
			*mutex = new_mutex;
		else
			/* Somebody already initialized it while we were
			 * calling sg_mutex_init, this one is now useless */
#ifdef STARPU_HAVE_SIMGRID_MUTEX_H
			sg_mutex_destroy(new_mutex);
#else
			xbt_mutex_destroy(new_mutex);
#endif
	}

#ifdef STARPU_HAVE_SIMGRID_MUTEX_H
	sg_mutex_lock(*mutex);
#else
	xbt_mutex_acquire(*mutex);
#endif

	_starpu_trace_mutex_locked();

	return 0;
}

int starpu_pthread_mutex_unlock(starpu_pthread_mutex_t *mutex)
{
	_starpu_trace_unlocking_mutex();

#ifdef STARPU_HAVE_SIMGRID_MUTEX_H
	sg_mutex_unlock(*mutex);
#else
	xbt_mutex_release(*mutex);
#endif

	_starpu_trace_mutex_unlocked();

	return 0;
}

int starpu_pthread_mutex_trylock(starpu_pthread_mutex_t *mutex)
{
	int ret;
	_starpu_trace_trylock_mutex();

#ifdef STARPU_HAVE_SIMGRID_MUTEX_H
	ret = sg_mutex_try_lock(*mutex);
#elif defined(HAVE_XBT_MUTEX_TRY_ACQUIRE) || defined(xbt_mutex_try_acquire)
	ret = xbt_mutex_try_acquire(*mutex);
#else
	ret = simcall_mutex_trylock((smx_mutex_t)*mutex);
#endif
	ret = ret ? 0 : EBUSY;

	if (ret)
	{
		_starpu_trace_mutex_locked();
	}
	return ret;
}

int starpu_pthread_mutexattr_gettype(const starpu_pthread_mutexattr_t *attr STARPU_ATTRIBUTE_UNUSED, int *type STARPU_ATTRIBUTE_UNUSED)
{
	return 0;
}

int starpu_pthread_mutexattr_settype(starpu_pthread_mutexattr_t *attr STARPU_ATTRIBUTE_UNUSED, int type STARPU_ATTRIBUTE_UNUSED)
{
	return 0;
}

int starpu_pthread_mutexattr_destroy(starpu_pthread_mutexattr_t *attr STARPU_ATTRIBUTE_UNUSED)
{
	return 0;
}

int starpu_pthread_mutexattr_init(starpu_pthread_mutexattr_t *attr STARPU_ATTRIBUTE_UNUSED)
{
	return 0;
}

/* Indexed by key-1 */
static int used_key[MAX_TSD];

int starpu_pthread_key_create(starpu_pthread_key_t *key, void (*destr_function) (void *) STARPU_ATTRIBUTE_UNUSED)
{
	unsigned i;

	/* Note: no synchronization here, we are actually monothreaded anyway. */
	for (i = 0; i < MAX_TSD; i++)
	{
		if (!used_key[i])
		{
			used_key[i] = 1;
			break;
		}
	}
	STARPU_ASSERT(i < MAX_TSD);
	/* key 0 is for process pointer argument */
	*key = i+1;
	return 0;
}

int starpu_pthread_key_delete(starpu_pthread_key_t key)
{
	used_key[key-1] = 0;
	return 0;
}

/* We need it only when using smpi */
#pragma weak smpi_process_get_user_data
#if !HAVE_DECL_SMPI_PROCESS_SET_USER_DATA && !defined(smpi_process_get_user_data)
extern void *smpi_process_get_user_data();
#endif

int starpu_pthread_setspecific(starpu_pthread_key_t key, const void *pointer)
{
	void **array;
#ifdef HAVE_SG_ACTOR_GET_DATA
	array = sg_actor_get_data(sg_actor_self());
#elif defined(HAVE_SG_ACTOR_DATA)
	array = sg_actor_data(sg_actor_self());
#else
#if defined(HAVE_SMPI_PROCESS_SET_USER_DATA) || defined(smpi_process_get_user_data)
#if defined(HAVE_MSG_PROCESS_SELF_NAME) || defined(MSG_process_self_name)
	const char *process_name = MSG_process_self_name();
#else
	const char *process_name = SIMIX_process_self_get_name();
#endif
	char *end;
	/* Test whether it is an MPI rank */
	strtol(process_name, &end, 10);
	if (!*end || !strcmp(process_name, "wait for mpi transfer") ||
			(!strcmp(process_name, "main") && _starpu_simgrid_running_smpi()))
		/* Special-case the SMPI process */
		array = smpi_process_get_user_data();
	else
#endif
		array = MSG_process_get_data(MSG_process_self());
#endif
	array[key] = (void*) pointer;
	return 0;
}

void* starpu_pthread_getspecific(starpu_pthread_key_t key)
{
	void **array;
#ifdef HAVE_SG_ACTOR_GET_DATA
	array = sg_actor_get_data(sg_actor_self());
#elif defined(HAVE_SG_ACTOR_DATA)
	array = sg_actor_data(sg_actor_self());
#else
#if defined(HAVE_SMPI_PROCESS_SET_USER_DATA) || defined(smpi_process_get_user_data)
#if defined(HAVE_MSG_PROCESS_SELF_NAME) || defined(MSG_process_self_name)
	const char *process_name = MSG_process_self_name();
#else
	const char *process_name = SIMIX_process_self_get_name();
#endif
	char *end;
	/* Test whether it is an MPI rank */
	strtol(process_name, &end, 10);
	if (!*end || !strcmp(process_name, "wait for mpi transfer") ||
			(!strcmp(process_name, "main") && _starpu_simgrid_running_smpi()))
		/* Special-case the SMPI processes */
		array = smpi_process_get_user_data();
	else
#endif
		array = MSG_process_get_data(MSG_process_self());
#endif
	if (!array)
		return NULL;
	return array[key];
}

int starpu_pthread_cond_init(starpu_pthread_cond_t *cond, starpu_pthread_condattr_t *cond_attr STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_HAVE_SIMGRID_COND_H
	*cond = sg_cond_init();
#else
	*cond = xbt_cond_init();
#endif
	return 0;
}

static void _starpu_pthread_cond_auto_init(starpu_pthread_cond_t *cond)
{
	/* Note: this is actually safe, because simgrid only preempts within
	 * simgrid functions */
	if (!*cond)
	{
		/* Here we may get preempted */
#ifdef STARPU_HAVE_SIMGRID_COND_H
		sg_cond_t new_cond = sg_cond_init();
#else
		xbt_cond_t new_cond = xbt_cond_init();
#endif
		if (!*cond)
			*cond = new_cond;
		else
			/* Somebody already initialized it while we were
			 * calling xbt_cond_init, this one is now useless */
#ifdef STARPU_HAVE_SIMGRID_COND_H
			sg_cond_destroy(new_cond);
#else
			xbt_cond_destroy(new_cond);
#endif
	}
}

int starpu_pthread_cond_signal(starpu_pthread_cond_t *cond)
{
	_starpu_pthread_cond_auto_init(cond);
#ifdef STARPU_HAVE_SIMGRID_COND_H
	sg_cond_notify_one(*cond);
#else
	xbt_cond_signal(*cond);
#endif
	return 0;
}

int starpu_pthread_cond_broadcast(starpu_pthread_cond_t *cond)
{
	_starpu_pthread_cond_auto_init(cond);
#ifdef STARPU_HAVE_SIMGRID_COND_H
	sg_cond_notify_all(*cond);
#else
	xbt_cond_broadcast(*cond);
#endif
	return 0;
}

int starpu_pthread_cond_wait(starpu_pthread_cond_t *cond, starpu_pthread_mutex_t *mutex)
{
	_starpu_trace_cond_wait_begin();

	_starpu_pthread_cond_auto_init(cond);
#ifdef STARPU_HAVE_SIMGRID_COND_H
	sg_cond_wait(*cond, *mutex);
#else
	xbt_cond_wait(*cond, *mutex);
#endif

	_starpu_trace_cond_wait_end();

	return 0;
}

int starpu_pthread_cond_timedwait(starpu_pthread_cond_t *cond, starpu_pthread_mutex_t *mutex, const struct timespec *abstime)
{
#if SIMGRID_VERSION >= 31800
	struct timespec now, delta;
	double delay;
	int ret = 0;
	_starpu_clock_gettime(&now);
	delta.tv_sec = abstime->tv_sec - now.tv_sec;
	delta.tv_nsec = abstime->tv_nsec - now.tv_nsec;
	delay = (double) delta.tv_sec + (double) delta.tv_nsec / 1000000000.;

	_starpu_trace_cond_wait_begin();

	_starpu_pthread_cond_auto_init(cond);
#ifdef STARPU_HAVE_SIMGRID_COND_H
	ret = sg_cond_wait_for(*cond, *mutex, delay) ? ETIMEDOUT : 0;
#else
	ret = xbt_cond_timedwait(*cond, *mutex, delay) ? ETIMEDOUT : 0;
#endif

	_starpu_trace_cond_wait_end();

	return ret;
#else
	STARPU_ASSERT_MSG(0, "simgrid version is too old for this");
#endif
}

int starpu_pthread_cond_destroy(starpu_pthread_cond_t *cond)
{
	if (*cond)
#ifdef STARPU_HAVE_SIMGRID_COND_H
		sg_cond_destroy(*cond);
#else
		xbt_cond_destroy(*cond);
#endif
	return 0;
}

/* TODO: use rwlocks
 * https://framagit.org/simgrid/simgrid/-/issues/92
 */
int starpu_pthread_rwlock_init(starpu_pthread_rwlock_t *restrict rwlock, const starpu_pthread_rwlockattr_t *restrict attr STARPU_ATTRIBUTE_UNUSED)
{
	return starpu_pthread_mutex_init(rwlock, NULL);
}

int starpu_pthread_rwlock_destroy(starpu_pthread_rwlock_t *rwlock)
{
	return starpu_pthread_mutex_destroy(rwlock);
}

int starpu_pthread_rwlock_rdlock(starpu_pthread_rwlock_t *rwlock)
{
	_starpu_trace_rdlocking_rwlock();

	int p_ret = starpu_pthread_mutex_lock(rwlock);

	_starpu_trace_rwlock_rdlocked();

	return p_ret;
}

int starpu_pthread_rwlock_tryrdlock(starpu_pthread_rwlock_t *rwlock)
{
	int p_ret = starpu_pthread_mutex_trylock(rwlock);

	if (!p_ret)
		_starpu_trace_rwlock_rdlocked();

	return p_ret;
}

int starpu_pthread_rwlock_wrlock(starpu_pthread_rwlock_t *rwlock)
{
	_starpu_trace_wrlocking_rwlock();

	int p_ret = starpu_pthread_mutex_lock(rwlock);

	_starpu_trace_rwlock_wrlocked();

	return p_ret;
}

int starpu_pthread_rwlock_trywrlock(starpu_pthread_rwlock_t *rwlock)
{
	int p_ret =  starpu_pthread_mutex_trylock(rwlock);

	if (!p_ret)
		_starpu_trace_rwlock_rdlocked();

	return p_ret;
}

int starpu_pthread_rwlock_unlock(starpu_pthread_rwlock_t *rwlock)
{
	_starpu_trace_unlocking_rwlock();

	int p_ret = starpu_pthread_mutex_unlock(rwlock);

	_starpu_trace_rwlock_unlocked();

	return p_ret;
}

#ifdef STARPU_HAVE_SIMGRID_BARRIER_H
int starpu_pthread_barrier_init(starpu_pthread_barrier_t *restrict barrier, const starpu_pthread_barrierattr_t *restrict attr STARPU_ATTRIBUTE_UNUSED, unsigned count)
{
	*barrier = sg_barrier_init(count);
	return 0;
}

int starpu_pthread_barrier_destroy(starpu_pthread_barrier_t *barrier)
{
	if (*barrier)
		sg_barrier_destroy(*barrier);
	return 0;
}

int starpu_pthread_barrier_wait(starpu_pthread_barrier_t *barrier)
{
	int ret;

	_starpu_trace_barrier_wait_begin();

	ret = sg_barrier_wait(*barrier);

	_starpu_trace_barrier_wait_end();
	return ret;
}
#elif defined(STARPU_SIMGRID_HAVE_XBT_BARRIER_INIT) || defined(xbt_barrier_init)
int starpu_pthread_barrier_init(starpu_pthread_barrier_t *restrict barrier, const starpu_pthread_barrierattr_t *restrict attr STARPU_ATTRIBUTE_UNUSED, unsigned count)
{
	*barrier = xbt_barrier_init(count);
	return 0;
}

int starpu_pthread_barrier_destroy(starpu_pthread_barrier_t *barrier)
{
	if (*barrier)
		xbt_barrier_destroy(*barrier);
	return 0;
}

int starpu_pthread_barrier_wait(starpu_pthread_barrier_t *barrier)
{
	int ret;

	_starpu_trace_barrier_wait_begin();

	ret = xbt_barrier_wait(*barrier);

	_starpu_trace_barrier_wait_end();
	return ret;
}
#endif /* defined(STARPU_SIMGRID_HAVE_XBT_BARRIER_INIT) */

int starpu_pthread_queue_init(starpu_pthread_queue_t *q)
{
	STARPU_PTHREAD_MUTEX_INIT(&q->mutex, NULL);
	q->queue = NULL;
	q->allocqueue = 0;
	q->nqueue = 0;
	return 0;
}

int starpu_pthread_wait_init(starpu_pthread_wait_t *w)
{
	STARPU_PTHREAD_MUTEX_INIT(&w->mutex, NULL);
	STARPU_PTHREAD_COND_INIT(&w->cond, NULL);
	w->block = 1;
	return 0;
}

int starpu_pthread_queue_register(starpu_pthread_wait_t *w, starpu_pthread_queue_t *q)
{
	STARPU_PTHREAD_MUTEX_LOCK(&q->mutex);

	if (q->nqueue == q->allocqueue)
	{
		/* Make room for the new waiter */
		unsigned newalloc;
		newalloc = q->allocqueue * 2;
		if (!newalloc)
			newalloc = 1;
		_STARPU_REALLOC(q->queue, newalloc * sizeof(*(q->queue)));
		q->allocqueue = newalloc;
	}
	q->queue[q->nqueue++] = w;

	STARPU_PTHREAD_MUTEX_UNLOCK(&q->mutex);
	return 0;
}

int starpu_pthread_queue_unregister(starpu_pthread_wait_t *w, starpu_pthread_queue_t *q)
{
	unsigned i;
	STARPU_PTHREAD_MUTEX_LOCK(&q->mutex);
	for (i = 0; i < q->nqueue; i++)
	{
		if (q->queue[i] == w)
		{
			memmove(&q->queue[i], &q->queue[i+1], (q->nqueue - i - 1) * sizeof(*(q->queue)));
			break;
		}
	}
	STARPU_ASSERT(i < q->nqueue);
	q->nqueue--;
	STARPU_PTHREAD_MUTEX_UNLOCK(&q->mutex);
	return 0;
}

int starpu_pthread_wait_reset(starpu_pthread_wait_t *w)
{
	STARPU_PTHREAD_MUTEX_LOCK(&w->mutex);
	w->block = 1;
	STARPU_PTHREAD_MUTEX_UNLOCK(&w->mutex);
	return 0;
}

int starpu_pthread_wait_wait(starpu_pthread_wait_t *w)
{
	STARPU_PTHREAD_MUTEX_LOCK(&w->mutex);
	while (w->block == 1)
		STARPU_PTHREAD_COND_WAIT(&w->cond, &w->mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK(&w->mutex);
	return 0;
}

/* pthread_cond_timedwait not yet available on windows, but we don't run simgrid there anyway */
#ifdef STARPU_SIMGRID
int starpu_pthread_wait_timedwait(starpu_pthread_wait_t *w, const struct timespec *abstime)
{
	STARPU_PTHREAD_MUTEX_LOCK(&w->mutex);
	while (w->block == 1)
		STARPU_PTHREAD_COND_TIMEDWAIT(&w->cond, &w->mutex, abstime);
	STARPU_PTHREAD_MUTEX_UNLOCK(&w->mutex);
	return 0;
}
#endif

int starpu_pthread_queue_signal(starpu_pthread_queue_t *q)
{
	starpu_pthread_wait_t *w;
	STARPU_PTHREAD_MUTEX_LOCK(&q->mutex);
	if (q->nqueue)
	{
		/* TODO: better try to wake a sleeping one if possible */
		w = q->queue[0];
		STARPU_PTHREAD_MUTEX_LOCK(&w->mutex);
		w->block = 0;
		STARPU_PTHREAD_COND_SIGNAL(&w->cond);
		STARPU_PTHREAD_MUTEX_UNLOCK(&w->mutex);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&q->mutex);
	return 0;
}

int starpu_pthread_queue_broadcast(starpu_pthread_queue_t *q)
{
	unsigned i;
	starpu_pthread_wait_t *w;
	STARPU_PTHREAD_MUTEX_LOCK(&q->mutex);
	for (i = 0; i < q->nqueue; i++)
	{
		w = q->queue[i];
		STARPU_PTHREAD_MUTEX_LOCK(&w->mutex);
		w->block = 0;
		STARPU_PTHREAD_COND_SIGNAL(&w->cond);
		STARPU_PTHREAD_MUTEX_UNLOCK(&w->mutex);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&q->mutex);
	return 0;
}

int starpu_pthread_wait_destroy(starpu_pthread_wait_t *w)
{
	STARPU_PTHREAD_MUTEX_LOCK(&w->mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK(&w->mutex);
	STARPU_PTHREAD_MUTEX_DESTROY(&w->mutex);
	STARPU_PTHREAD_COND_DESTROY(&w->cond);
	return 0;
}

int starpu_pthread_queue_destroy(starpu_pthread_queue_t *q)
{
	STARPU_ASSERT(!q->nqueue);
	STARPU_PTHREAD_MUTEX_LOCK(&q->mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK(&q->mutex);
	STARPU_PTHREAD_MUTEX_DESTROY(&q->mutex);
	free(q->queue);
	return 0;
}

#endif /* STARPU_SIMGRID */

#if (defined(STARPU_SIMGRID) && !defined(STARPU_HAVE_SIMGRID_BARRIER_H) && !defined(STARPU_SIMGRID_HAVE_XBT_BARRIER_INIT) && !defined(xbt_barrier_init)) || (!defined(STARPU_SIMGRID) && !defined(STARPU_HAVE_PTHREAD_BARRIER))
int starpu_pthread_barrier_init(starpu_pthread_barrier_t *restrict barrier, const starpu_pthread_barrierattr_t *restrict attr STARPU_ATTRIBUTE_UNUSED, unsigned count)
{
	int ret = starpu_pthread_mutex_init(&barrier->mutex, NULL);
	if (!ret)
		ret = starpu_pthread_cond_init(&barrier->cond, NULL);
	if (!ret)
		ret = starpu_pthread_cond_init(&barrier->cond_destroy, NULL);
	barrier->count = count;
	barrier->done = 0;
	barrier->busy = 0;
	return ret;
}

int starpu_pthread_barrier_destroy(starpu_pthread_barrier_t *barrier)
{
	starpu_pthread_mutex_lock(&barrier->mutex);
	while (barrier->busy)
	{
		starpu_pthread_cond_wait(&barrier->cond_destroy, &barrier->mutex);
	}
	starpu_pthread_mutex_unlock(&barrier->mutex);
	int ret = starpu_pthread_mutex_destroy(&barrier->mutex);
	if (!ret)
		ret = starpu_pthread_cond_destroy(&barrier->cond);
	if (!ret)
		ret = starpu_pthread_cond_destroy(&barrier->cond_destroy);
	return ret;
}

int starpu_pthread_barrier_wait(starpu_pthread_barrier_t *barrier)
{
	int ret = 0;
	_starpu_trace_barrier_wait_begin();

	starpu_pthread_mutex_lock(&barrier->mutex);
	barrier->done++;
	if (barrier->done == barrier->count)
	{
		barrier->done = 0;
		starpu_pthread_cond_broadcast(&barrier->cond);
		ret = STARPU_PTHREAD_BARRIER_SERIAL_THREAD;
	}
	else
	{
		barrier->busy++;
		starpu_pthread_cond_wait(&barrier->cond, &barrier->mutex);
		barrier->busy--;
		starpu_pthread_cond_broadcast(&barrier->cond_destroy);
	}

	starpu_pthread_mutex_unlock(&barrier->mutex);

	_starpu_trace_barrier_wait_end();

	return ret;
}
#endif /* defined(STARPU_SIMGRID) || !defined(STARPU_HAVE_PTHREAD_BARRIER) */

#ifdef STARPU_FXT_LOCK_TRACES
#if !defined(STARPU_SIMGRID) && !defined(_MSC_VER) /* !STARPU_SIMGRID */
int starpu_pthread_mutex_lock(starpu_pthread_mutex_t *mutex)
{
	_starpu_trace_locking_mutex();

	int p_ret = pthread_mutex_lock(mutex);

	_starpu_trace_mutex_locked();

	return p_ret;
}

int starpu_pthread_mutex_unlock(starpu_pthread_mutex_t *mutex)
{
	_starpu_trace_unlocking_mutex();

	int p_ret = pthread_mutex_unlock(mutex);

	_starpu_trace_mutex_unlocked();

	return p_ret;
}

int starpu_pthread_mutex_trylock(starpu_pthread_mutex_t *mutex)
{
	int ret;
	_starpu_trace_trylock_mutex();

	ret = pthread_mutex_trylock(mutex);

	if (!ret)
		_starpu_trace_mutex_locked();

	return ret;
}

int starpu_pthread_cond_wait(starpu_pthread_cond_t *cond, starpu_pthread_mutex_t *mutex)
{
	_starpu_trace_cond_wait_begin();

	int p_ret = pthread_cond_wait(cond, mutex);

	_starpu_trace_cond_wait_end();

	return p_ret;
}

int starpu_pthread_rwlock_rdlock(starpu_pthread_rwlock_t *rwlock)
{
	_starpu_trace_rdlocking_rwlock();

	int p_ret = pthread_rwlock_rdlock(rwlock);

	_starpu_trace_rwlock_rdlocked();

	return p_ret;
}

int starpu_pthread_rwlock_tryrdlock(starpu_pthread_rwlock_t *rwlock)
{
	_starpu_trace_rdlocking_rwlock();

	int p_ret = pthread_rwlock_tryrdlock(rwlock);

	if (!p_ret)
		_starpu_trace_rwlock_rdlocked();

	return p_ret;
}

int starpu_pthread_rwlock_wrlock(starpu_pthread_rwlock_t *rwlock)
{
	_starpu_trace_wrlocking_rwlock();

	int p_ret = pthread_rwlock_wrlock(rwlock);

	_starpu_trace_rwlock_wrlocked();

	return p_ret;
}

int starpu_pthread_rwlock_trywrlock(starpu_pthread_rwlock_t *rwlock)
{
	_starpu_trace_wrlocking_rwlock();

	int p_ret = pthread_rwlock_trywrlock(rwlock);

	if (!p_ret)
		_starpu_trace_rwlock_wrlocked();

	return p_ret;
}

int starpu_pthread_rwlock_unlock(starpu_pthread_rwlock_t *rwlock)
{
	_starpu_trace_unlocking_rwlock();

	int p_ret = pthread_rwlock_unlock(rwlock);

	_starpu_trace_rwlock_unlocked();

	return p_ret;
}
#endif /* !defined(STARPU_SIMGRID) && !defined(_MSC_VER) */

#if !defined(STARPU_SIMGRID) && !defined(_MSC_VER) && defined(STARPU_HAVE_PTHREAD_BARRIER)
int starpu_pthread_barrier_wait(starpu_pthread_barrier_t *barrier)
{
	int ret;
	_starpu_trace_barrier_wait_begin();

	ret = pthread_barrier_wait(barrier);

	_starpu_trace_barrier_wait_end();

	return ret;
}
#endif /* STARPU_SIMGRID, _MSC_VER, STARPU_HAVE_PTHREAD_BARRIER */

#endif /* STARPU_FXT_LOCK_TRACES */

/* "sched" variants, to be used (through the STARPU_PTHREAD_MUTEX_*LOCK_SCHED
 * macros of course) which record when the mutex is held or not */
int starpu_pthread_mutex_lock_sched(starpu_pthread_mutex_t *mutex)
{
	return starpu_pthread_mutex_lock(mutex);
}

int starpu_pthread_mutex_unlock_sched(starpu_pthread_mutex_t *mutex)
{
	return starpu_pthread_mutex_unlock(mutex);
}

int starpu_pthread_mutex_trylock_sched(starpu_pthread_mutex_t *mutex)
{
	return starpu_pthread_mutex_trylock(mutex);
}

#ifdef STARPU_DEBUG
void starpu_pthread_mutex_check_sched(starpu_pthread_mutex_t *mutex, char *file, int line)
{
	int workerid = starpu_worker_get_id();
	STARPU_ASSERT_MSG(workerid == -1 || !_starpu_worker_mutex_is_sched_mutex(workerid, mutex), "%s:%d is locking/unlocking a sched mutex but not using STARPU_PTHREAD_MUTEX_LOCK_SCHED", file, line);
}
#endif

#if defined(STARPU_SIMGRID) || (defined(STARPU_LINUX_SYS) && defined(STARPU_HAVE_XCHG)) || !defined(HAVE_PTHREAD_SPIN_LOCK)

#undef starpu_pthread_spin_init
int starpu_pthread_spin_init(starpu_pthread_spinlock_t *lock, int pshared)
{
	return _starpu_pthread_spin_init(lock, pshared);
}

#undef starpu_pthread_spin_destroy
int starpu_pthread_spin_destroy(starpu_pthread_spinlock_t *lock STARPU_ATTRIBUTE_UNUSED)
{
	return _starpu_pthread_spin_destroy(lock);
}

#undef starpu_pthread_spin_lock
int starpu_pthread_spin_lock(starpu_pthread_spinlock_t *lock)
{
	return _starpu_pthread_spin_lock(lock);
}
#endif

#if defined(STARPU_SIMGRID) || (defined(STARPU_LINUX_SYS) && defined(STARPU_HAVE_XCHG)) || !defined(STARPU_HAVE_PTHREAD_SPIN_LOCK)

#if !defined(STARPU_SIMGRID) && defined(STARPU_LINUX_SYS) && defined(STARPU_HAVE_XCHG)
int _starpu_pthread_spin_do_lock(starpu_pthread_spinlock_t *lock)
{
	if (STARPU_VAL_COMPARE_AND_SWAP(&lock->taken, 0, 1) == 0)
		/* Got it on first try! */
		return 0;

	/* Busy, spin a bit.  */
	unsigned i;
	for (i = 0; i < 128; i++)
	{
		/* Pause a bit before retrying */
		STARPU_UYIELD();
		/* And synchronize with other threads */
		STARPU_SYNCHRONIZE();
		if (!lock->taken)
			/* Holder released it, try again */
			if (STARPU_VAL_COMPARE_AND_SWAP(&lock->taken, 0, 1) == 0)
				/* Got it! */
				return 0;
	}

	/* We have spent enough time with spinning, let's block */
	/* This avoids typical 10ms pauses when the application thread tries to submit tasks. */
	while (1)
	{
		/* Tell releaser to wake us */
		unsigned prev = STARPU_VAL_EXCHANGE(&lock->taken, 2);
		if (prev == 0)
			/* Ah, it just got released and we actually acquired
			 * it!
			 * Note: the sad thing is that we have just written 2,
			 * so will spuriously try to wake a thread on unlock,
			 * but we can not avoid it since we do not know whether
			 * there are other threads sleeping or not.
			 */
			return 0;

		/* Now start sleeping (unless it was released in between)
		 * We are sure to get woken because either
		 * - some thread has not released the lock yet, and lock->taken
		 *   is 2, so it will wake us.
		 * - some other thread started blocking, and will set
		 *   lock->taken back to 2
		 */
#ifdef _STARPU_SPIN_LITHE
		/* Lithe port (2026-10-23): a Lithe context parks on liblithe's futex
		 * instead of sleeping its hart in the kernel; unlock wakes both. */
		if (_starpu_lithe_coop())
		{
			starpu_lithe_futex_wait_((int *) &lock->taken, 2);
			continue;
		}
#endif
		if (syscall(SYS_futex, &lock->taken, _starpu_futex_wait, 2, NULL, NULL, 0))
			if (errno == ENOSYS)
				_starpu_futex_wait = FUTEX_WAIT;
	}
}
#endif

#undef starpu_pthread_spin_trylock
int starpu_pthread_spin_trylock(starpu_pthread_spinlock_t *lock)
{
	return _starpu_pthread_spin_trylock(lock);
}

#undef starpu_pthread_spin_unlock
int starpu_pthread_spin_unlock(starpu_pthread_spinlock_t *lock)
{
	return _starpu_pthread_spin_unlock(lock);
}

#if !defined(STARPU_SIMGRID) && defined(STARPU_LINUX_SYS) && defined(STARPU_HAVE_XCHG)
void _starpu_pthread_spin_do_unlock(starpu_pthread_spinlock_t *lock)
{
	/*
	 * Somebody to wake. Clear 'taken' and wake him.
	 * Note that he may not be sleeping yet, but if he is not, we won't
	 * since the value of 'taken' will have changed.
	 */
	lock->taken = 0;
	STARPU_SYNCHRONIZE();
#ifdef _STARPU_SPIN_LITHE
	if (starpu_lithe_futex_wake_some_)
		starpu_lithe_futex_wake_some_((int *) &lock->taken, 1);
#endif
	if (syscall(SYS_futex, &lock->taken, _starpu_futex_wake, 1, NULL, NULL, 0) == -1)
		switch (errno)
		{
			case ENOSYS:
				_starpu_futex_wake = FUTEX_WAKE;
				if (syscall(SYS_futex, &lock->taken, _starpu_futex_wake, 1, NULL, NULL, 0) == -1)
					STARPU_ASSERT_MSG(0, "futex(wake) returned %d!", errno);
				break;
			case 0:
				break;
			default:
				STARPU_ASSERT_MSG(0, "futex returned %d!", errno);
				break;
		}
}

#endif

#endif /* defined(STARPU_SIMGRID) || (defined(STARPU_LINUX_SYS) && defined(STARPU_HAVE_XCHG)) || !defined(STARPU_HAVE_PTHREAD_SPIN_LOCK) */

#ifdef STARPU_SIMGRID

int starpu_sem_destroy(starpu_sem_t *sem)
{
#ifdef STARPU_HAVE_SIMGRID_SEMAPHORE_H
	sg_sem_destroy(*sem);
#else
	MSG_sem_destroy(*sem);
#endif
	return 0;
}

int starpu_sem_init(starpu_sem_t *sem, int pshared, unsigned value)
{
	STARPU_ASSERT_MSG(pshared == 0, "pshared semaphores not supported under simgrid");
#ifdef STARPU_HAVE_SIMGRID_SEMAPHORE_H
	*sem = sg_sem_init(value);
#else
	*sem = MSG_sem_init(value);
#endif
	return 0;
}

int starpu_sem_post(starpu_sem_t *sem)
{
#ifdef STARPU_HAVE_SIMGRID_SEMAPHORE_H
	sg_sem_release(*sem);
#else
	MSG_sem_release(*sem);
#endif
	return 0;
}

int starpu_sem_wait(starpu_sem_t *sem)
{
#ifdef STARPU_HAVE_SIMGRID_SEMAPHORE_H
	sg_sem_acquire(*sem);
#else
	MSG_sem_acquire(*sem);
#endif
	return 0;
}

int starpu_sem_trywait(starpu_sem_t *sem)
{
#ifdef STARPU_HAVE_SIMGRID_SEMAPHORE_H
	if (sg_sem_would_block(*sem))
#else
	if (MSG_sem_would_block(*sem))
#endif
		return EAGAIN;
	starpu_sem_wait(sem);
	return 0;
}

int starpu_sem_getvalue(starpu_sem_t *sem, int *sval)
{
#if SIMGRID_VERSION > 31300
#  ifdef STARPU_HAVE_SIMGRID_SEMAPHORE_H
	*sval = sg_sem_get_capacity(*sem);
#  else
	*sval = MSG_sem_get_capacity(*sem);
#  endif
	return 0;
#else
	(void) sem;
	(void) sval;
	STARPU_ABORT_MSG("sigmrid up to 3.13 did not have working MSG_sem_get_capacity");
#endif
}

#elif !defined(_MSC_VER) || defined(BUILDING_STARPU) /* !STARPU_SIMGRID */

#ifdef STARPU_LITHE
extern int lithe_can_block(void) __attribute__((weak));
extern void lithe_context_yield(void) __attribute__((weak));
#endif
int starpu_sem_wait(starpu_sem_t *sem)
{
	int ret;
#ifdef STARPU_LITHE
	/* Lithe port: a context must not block its hart in sem_wait; try and yield. */
	if (lithe_can_block && lithe_context_yield && lithe_can_block() && !getenv("STARPU_LITHE_DISABLE"))
	{
		while ((ret = sem_trywait(sem)) == -1 && (errno == EAGAIN || errno == EINTR))
			lithe_context_yield();
		return ret;
	}
#endif
	while((ret = sem_wait(sem)) == -1 && errno == EINTR)
		;

	return ret;
}

int starpu_sem_trywait(starpu_sem_t *sem)
{
	int ret;
	while((ret = sem_trywait(sem)) == -1 && errno == EINTR)
		;
	
	return ret;
}

#endif

#if defined(STARPU_LITHE) && !defined(STARPU_FXT_LOCK_TRACES) && !defined(STARPU_SIMGRID) && !defined(_MSC_VER)
/* ---------------------------------------------------------------------------
 * Lithe port (lithe-docker, 2026-10-19). StarPU's workers become Lithe fork-join
 * contexts in the caller's scheduler (daemon: they live as long as StarPU), and
 * every StarPU wait -- mutex, rwlock, condition variable, semaphore -- lets the
 * hart run other contexts instead of blocking it in the kernel: a Lithe context
 * tries and yields, or parks on liblithe's futex; anything else (a kernel thread,
 * a context that cannot block) uses pthreads unchanged. The pthread types stay
 * (static initialisers); condition variables get a side table of sequence words
 * keyed by address. All Lithe symbols are weak: without liblithe, or with
 * STARPU_LITHE_DISABLE=1, this is stock StarPU. Set STARPU_WORKERS_NOBIND=1 under
 * Lithe so workers do not pin the hart threads.
 * ------------------------------------------------------------------------- */
#include <sched.h>
#include <errno.h>
#include <time.h>
typedef struct lithe_sched lithe_sched_t;
typedef struct lithe_fork_join_sched lithe_fork_join_sched_t;
typedef struct lithe_fork_join_context lithe_fork_join_context_t;
extern lithe_sched_t *lithe_sched_current(void) __attribute__((weak));
extern int lithe_can_block(void) __attribute__((weak));
extern void lithe_context_yield(void) __attribute__((weak));
extern lithe_fork_join_sched_t *lithe_ensure_root_fork_join_sched(void) __attribute__((weak));
extern lithe_fork_join_context_t *lithe_fork_join_context_create(lithe_fork_join_sched_t *, size_t, void (*)(void *), void *) __attribute__((weak));
extern void lithe_fork_join_context_set_daemon(lithe_fork_join_context_t *) __attribute__((weak));
extern const char lithe_fork_join_sched_funcs[] __attribute__((weak));
extern int starpu_lithe_futex_wait_(int *, int) __asm__("futex_wait") __attribute__((weak));
extern int starpu_lithe_futex_wake_some_(int *, int) __asm__("futex_wake_some") __attribute__((weak));

static int _starpu_lithe_on(void)
{
	static int on = -1;
	if (on < 0)
	{
		const char *d = getenv("STARPU_LITHE_DISABLE");
		on = !(d && d[0] == '1') && lithe_can_block && lithe_context_yield && lithe_sched_current
		     && lithe_fork_join_context_create && starpu_lithe_futex_wait_ && starpu_lithe_futex_wake_some_;
	}
	return on;
}
/* The caller is a Lithe context that may yield or park. */
static int _starpu_lithe_coop(void);
void _starpu_lithe_uyield(void)
{
	static __thread unsigned n;
	_STARPU_UYIELD();
	if ((++n & 15) == 0 && _starpu_lithe_coop())
		lithe_context_yield();
}

static int _starpu_lithe_coop(void)
{
	return _starpu_lithe_on() && lithe_can_block();
}

/* ---- threads ---- */
struct _starpu_lithe_thr { void *(*fn)(void *); void *arg; void *ret; int done; int refs; };
static int _starpu_lithe_handle(pthread_t t) { return ((uintptr_t) t & 1u) != 0; }
static struct _starpu_lithe_thr *_starpu_lithe_rec(pthread_t t) { return (struct _starpu_lithe_thr *) ((uintptr_t) t & ~(uintptr_t) 1); }
static void _starpu_lithe_drop(struct _starpu_lithe_thr *r) { if (__atomic_sub_fetch(&r->refs, 1, __ATOMIC_ACQ_REL) == 0) free(r); }
static void _starpu_lithe_start(void *x)
{
	struct _starpu_lithe_thr *r = x;
	r->ret = r->fn(r->arg);
	__atomic_store_n(&r->done, 1, __ATOMIC_RELEASE);
	starpu_lithe_futex_wake_some_(&r->done, 1 << 30);
	_starpu_lithe_drop(r);
}
static lithe_fork_join_sched_t *_starpu_lithe_worker_sched(void)
{
	lithe_sched_t *cur = lithe_sched_current();
	if (cur && lithe_fork_join_sched_funcs && *(const void *const *) cur == (const void *) lithe_fork_join_sched_funcs)
		return (lithe_fork_join_sched_t *) cur;
	return lithe_ensure_root_fork_join_sched ? lithe_ensure_root_fork_join_sched() : NULL;
}
int starpu_pthread_create(starpu_pthread_t *thread, const starpu_pthread_attr_t *attr, void *(*start_routine)(void *), void *arg)
{
	if (_starpu_lithe_on() && lithe_sched_current() != NULL)
	{
		lithe_fork_join_sched_t *s = _starpu_lithe_worker_sched();
		struct _starpu_lithe_thr *r = calloc(1, sizeof(*r));
		size_t ss = 4u << 20;
		if (attr) { size_t a = 0; if (pthread_attr_getstacksize(attr, &a) == 0 && a > ss) ss = a; }
		if (s && r)
		{
			r->fn = start_routine; r->arg = arg; r->refs = 2;
			lithe_fork_join_context_t *c = lithe_fork_join_context_create(s, ss, _starpu_lithe_start, r);
			if (c)
			{
				if (lithe_fork_join_context_set_daemon) lithe_fork_join_context_set_daemon(c);
				*thread = (pthread_t) ((uintptr_t) r | 1u);
				return 0;
			}
		}
		free(r);
	}
	return pthread_create(thread, attr, start_routine, arg);
}
int starpu_pthread_join(starpu_pthread_t thread, void **retval)
{
	if (!_starpu_lithe_handle(thread))
		return pthread_join(thread, retval);
	struct _starpu_lithe_thr *r = _starpu_lithe_rec(thread);
	while (!__atomic_load_n(&r->done, __ATOMIC_ACQUIRE))
	{
		if (_starpu_lithe_coop()) starpu_lithe_futex_wait_(&r->done, 0);
		else sched_yield();
	}
	if (retval) *retval = r->ret;
	_starpu_lithe_drop(r);
	return 0;
}
int starpu_pthread_detach(starpu_pthread_t thread)
{
	if (!_starpu_lithe_handle(thread))
		return pthread_detach(thread);
	_starpu_lithe_drop(_starpu_lithe_rec(thread));
	return 0;
}

/* ---- mutexes and rwlocks: try, then yield the context ---- */
int starpu_pthread_mutex_lock(starpu_pthread_mutex_t *mutex)
{
	if (!_starpu_lithe_coop())
		return pthread_mutex_lock(mutex);
	int r;
	while ((r = pthread_mutex_trylock(mutex)) == EBUSY)
		lithe_context_yield();
	return r;
}
int starpu_pthread_mutex_unlock(starpu_pthread_mutex_t *mutex) { return pthread_mutex_unlock(mutex); }
int starpu_pthread_mutex_trylock(starpu_pthread_mutex_t *mutex) { return pthread_mutex_trylock(mutex); }
int starpu_pthread_rwlock_rdlock(starpu_pthread_rwlock_t *l)
{
	if (!_starpu_lithe_coop()) return pthread_rwlock_rdlock(l);
	int r; while ((r = pthread_rwlock_tryrdlock(l)) == EBUSY) lithe_context_yield(); return r;
}
int starpu_pthread_rwlock_wrlock(starpu_pthread_rwlock_t *l)
{
	if (!_starpu_lithe_coop()) return pthread_rwlock_wrlock(l);
	int r; while ((r = pthread_rwlock_trywrlock(l)) == EBUSY) lithe_context_yield(); return r;
}
int starpu_pthread_rwlock_tryrdlock(starpu_pthread_rwlock_t *l) { return pthread_rwlock_tryrdlock(l); }
int starpu_pthread_rwlock_trywrlock(starpu_pthread_rwlock_t *l) { return pthread_rwlock_trywrlock(l); }
int starpu_pthread_rwlock_unlock(starpu_pthread_rwlock_t *l) { return pthread_rwlock_unlock(l); }
#ifdef STARPU_HAVE_PTHREAD_BARRIER
/* ---- barrier: generation word parked on liblithe's futex (2026-10-23) ----
 * pthread_barrier_wait slept the hart in the kernel: a combined-worker
 * (parallel) task's members on fewer harts than members never all arrived.
 * When the port is on, every waiter uses this barrier: a Lithe context parks on
 * liblithe's futex, anything else on the kernel futex; the last arriver bumps
 * the generation and wakes both. State lives in a side table keyed by address
 * (the pthread type stays for the stock path). */
#define _STARPU_LITHE_BAR_SLOTS 256
static struct { void *key; unsigned count; unsigned arrived; int gen; int kwait; } _starpu_lithe_bar[_STARPU_LITHE_BAR_SLOTS];
static int _starpu_lithe_bar_slot(void *b, int create)
{
	uintptr_t h = ((uintptr_t) b >> 4) * 0x9E3779B97F4A7C15ull;
	for (unsigned i = 0; i < _STARPU_LITHE_BAR_SLOTS; i++)
	{
		unsigned k = (unsigned) ((h >> 40) + i) % _STARPU_LITHE_BAR_SLOTS;
		void *cur = __atomic_load_n(&_starpu_lithe_bar[k].key, __ATOMIC_ACQUIRE);
		if (cur == b) return (int) k;
		if (cur == NULL)
		{
			if (!create) return -1;
			void *z = NULL;
			if (__atomic_compare_exchange_n(&_starpu_lithe_bar[k].key, &z, b, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
				return (int) k;
			if (z == b) return (int) k;
		}
	}
	return -1;
}
int starpu_pthread_barrier_init(starpu_pthread_barrier_t *b, const starpu_pthread_barrierattr_t *attr, unsigned count)
{
	int k = _starpu_lithe_on() ? _starpu_lithe_bar_slot(b, 1) : -1;
	if (k >= 0)
	{
		_starpu_lithe_bar[k].count = count;
		__atomic_store_n(&_starpu_lithe_bar[k].arrived, 0, __ATOMIC_RELEASE);
	}
	return pthread_barrier_init(b, attr, count);
}
int starpu_pthread_barrier_destroy(starpu_pthread_barrier_t *b)
{
	int k = _starpu_lithe_bar_slot(b, 0);
	if (k >= 0)
		__atomic_store_n(&_starpu_lithe_bar[k].key, NULL, __ATOMIC_RELEASE);
	return pthread_barrier_destroy(b);
}
int starpu_pthread_barrier_wait(starpu_pthread_barrier_t *b)
{
	int k = _starpu_lithe_bar_slot(b, 0);
	if (k < 0) return pthread_barrier_wait(b);   /* port off, or table full at init */
	int g = __atomic_load_n(&_starpu_lithe_bar[k].gen, __ATOMIC_ACQUIRE);
	if (__atomic_add_fetch(&_starpu_lithe_bar[k].arrived, 1, __ATOMIC_ACQ_REL) == _starpu_lithe_bar[k].count)
	{
		__atomic_store_n(&_starpu_lithe_bar[k].arrived, 0, __ATOMIC_RELAXED);
		__atomic_add_fetch(&_starpu_lithe_bar[k].gen, 1, __ATOMIC_ACQ_REL);
		starpu_lithe_futex_wake_some_(&_starpu_lithe_bar[k].gen, 1 << 30);
		if (__atomic_load_n(&_starpu_lithe_bar[k].kwait, __ATOMIC_ACQUIRE) > 0)
			syscall(SYS_futex, &_starpu_lithe_bar[k].gen, FUTEX_WAKE_PRIVATE, 1 << 30, NULL, NULL, 0);
		return PTHREAD_BARRIER_SERIAL_THREAD;
	}
	while (__atomic_load_n(&_starpu_lithe_bar[k].gen, __ATOMIC_ACQUIRE) == g)
	{
		if (_starpu_lithe_coop())
			starpu_lithe_futex_wait_(&_starpu_lithe_bar[k].gen, g);
		else
		{
			__atomic_add_fetch(&_starpu_lithe_bar[k].kwait, 1, __ATOMIC_ACQ_REL);
			syscall(SYS_futex, &_starpu_lithe_bar[k].gen, FUTEX_WAIT_PRIVATE, g, NULL, NULL, 0);
			__atomic_sub_fetch(&_starpu_lithe_bar[k].kwait, 1, __ATOMIC_ACQ_REL);
		}
	}
	return 0;
}
#endif

/* ---- condition variables: sequence word per cond, parked on liblithe's futex ---- */
#define _STARPU_LITHE_CV_SLOTS 4096
static struct { void *key; int seq; } _starpu_lithe_cv[_STARPU_LITHE_CV_SLOTS];
static int *_starpu_lithe_cv_seq(void *cond)
{
	uintptr_t h = ((uintptr_t) cond >> 4) * 0x9E3779B97F4A7C15ull;
	for (unsigned i = 0; i < _STARPU_LITHE_CV_SLOTS; i++)
	{
		unsigned k = (unsigned) ((h >> 40) + i) % _STARPU_LITHE_CV_SLOTS;
		void *cur = __atomic_load_n(&_starpu_lithe_cv[k].key, __ATOMIC_ACQUIRE);
		if (cur == cond) return &_starpu_lithe_cv[k].seq;
		if (cur == NULL)
		{
			void *exp = NULL;
			if (__atomic_compare_exchange_n(&_starpu_lithe_cv[k].key, &exp, cond, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE) || exp == cond)
				return &_starpu_lithe_cv[k].seq;
		}
	}
	return NULL; /* table full: callers fall back to pthreads */
}
static void _starpu_lithe_cv_wake(void *cond, int n)
{
	if (!_starpu_lithe_on()) return;
	int *seq = _starpu_lithe_cv_seq(cond);
	if (!seq) return;
	__atomic_add_fetch(seq, 1, __ATOMIC_ACQ_REL);
	starpu_lithe_futex_wake_some_(seq, n);
}
int starpu_pthread_cond_signal(starpu_pthread_cond_t *cond)
{
	_starpu_lithe_cv_wake(cond, 1);
	return pthread_cond_signal(cond);
}
int starpu_pthread_cond_broadcast(starpu_pthread_cond_t *cond)
{
	_starpu_lithe_cv_wake(cond, 1 << 30);
	return pthread_cond_broadcast(cond);
}
int starpu_pthread_cond_wait(starpu_pthread_cond_t *cond, starpu_pthread_mutex_t *mutex)
{
	int *seq;
	if (!_starpu_lithe_coop() || !(seq = _starpu_lithe_cv_seq(cond)))
		return pthread_cond_wait(cond, mutex);
	int v = __atomic_load_n(seq, __ATOMIC_ACQUIRE); /* read under the mutex: a signal after it changes seq */
	pthread_mutex_unlock(mutex);
	starpu_lithe_futex_wait_(seq, v);
	return starpu_pthread_mutex_lock(mutex);
}
int starpu_pthread_cond_timedwait(starpu_pthread_cond_t *cond, starpu_pthread_mutex_t *mutex, const struct timespec *abstime)
{
	int *seq;
	if (!_starpu_lithe_coop() || !(seq = _starpu_lithe_cv_seq(cond)))
		return pthread_cond_timedwait(cond, mutex, abstime);
	int v = __atomic_load_n(seq, __ATOMIC_ACQUIRE), r = 0;
	pthread_mutex_unlock(mutex);
	for (;;)
	{
		if (__atomic_load_n(seq, __ATOMIC_ACQUIRE) != v) break;
		struct timespec now; clock_gettime(CLOCK_REALTIME, &now);
		if (now.tv_sec > abstime->tv_sec || (now.tv_sec == abstime->tv_sec && now.tv_nsec >= abstime->tv_nsec)) { r = ETIMEDOUT; break; }
		lithe_context_yield();
	}
	starpu_pthread_mutex_lock(mutex);
	return r;
}
#endif /* STARPU_LITHE */
