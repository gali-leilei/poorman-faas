package reaper

import (
	"container/heap"
	"context"
	"log/slog"
	"sync"
	"time"
)

// Expirer is an interface that defines the methods for expiring resources.
type Expirer interface {
	// Update updates the last accessed time of the resource.
	Update(ctx context.Context, uuid string) error
	// Expire returns a list of resources that have expired.
	Expire(ctx context.Context) []string
}

// PQExpirer is an expirer that uses a priority queue to expire resources.
type PQExpirer struct {
	TimeToLive time.Duration
	// mutex for the items map and the pq
	mu    sync.RWMutex
	pq    priorityQueue
	items map[string]*item // uuid -> item mapping for O(1) lookup
}

// NewPQExpirer creates a new PQExpirer with the given expiration time.
func NewPQExpirer(timeToLive time.Duration) *PQExpirer {
	pqe := &PQExpirer{
		pq:         make(priorityQueue, 0),
		items:      make(map[string]*item),
		TimeToLive: timeToLive,
	}
	heap.Init(&pqe.pq)
	return pqe
}

// Update updates the last accessed time of the resource.
// If the resource doesn't exist, it will be added to the priority queue.
func (e *PQExpirer) Update(ctx context.Context, uuid string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	now := time.Now()

	if existingItem, exists := e.items[uuid]; exists {
		// Update existing item's last access time
		existingItem.lastAccess = now
		heap.Fix(&e.pq, existingItem.index)
	} else {
		// Add new item
		newItem := &item{
			uuid:       uuid,
			lastAccess: now,
		}
		heap.Push(&e.pq, newItem)
		e.items[uuid] = newItem
	}

	return nil
}

// Expire returns a list of resources that have expired.
// Resources are considered expired if their last access time is older than the expiration time.
func (e *PQExpirer) Expire(ctx context.Context) []string {
	e.mu.Lock()
	defer e.mu.Unlock()

	now := time.Now()
	expired := make([]string, 0)

	// Pop items from the priority queue until we find one that hasn't expired
	for e.pq.Len() > 0 {
		// Peek at the oldest item
		oldest := e.pq[0]

		// Check if it has expired
		if now.Sub(oldest.lastAccess) >= e.TimeToLive {
			// Pop and collect expired item
			slog.Info("expiring item", "uuid", oldest.uuid, "lastAccess", oldest.lastAccess, "now", now,
				"expiredDuration", now.Sub(oldest.lastAccess), "timeToLive", e.TimeToLive)
			item := heap.Pop(&e.pq).(*item)
			delete(e.items, item.uuid)
			expired = append(expired, item.uuid)
		} else {
			// Since the queue is sorted by time, if this item hasn't expired,
			// no other items have expired either
			break
		}
	}

	return expired
}
