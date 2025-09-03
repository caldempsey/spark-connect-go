package types

import (
	"context"
	"errors"
	"io"
	"iter"
	"sync/atomic"

	"github.com/apache/arrow-go/v18/arrow"
)

type RowPull2 = iter.Seq2[Row, error]

// NewRowSequence flattens record batches to a sequence of rows stream.
func NewRowSequence(ctx context.Context, recordSeq iter.Seq2[arrow.Record, error]) iter.Seq2[Row, error] {
	return func(yield func(Row, error) bool) {
		for rec, recErr := range recordSeq {
			select {
			case <-ctx.Done():
				_ = yield(nil, ctx.Err())
				return
			default:
			}
			if recErr != nil {
				// forward upstream error once, then stop
				_ = yield(nil, recErr)
				return
			}
			if rec == nil {
				_ = yield(nil, errors.New("expected arrow.Record to contain non-nil Rows, got nil"))
				return
			}

			rows, err := func() ([]Row, error) {
				defer rec.Release()
				return ReadArrowRecordToRows(rec)
			}()
			if err != nil {
				_ = yield(nil, err)
				return
			}
			for _, row := range rows {
				if !yield(row, nil) {
					return
				}
			}
		}
	}
}

// NewRowPull2 iterates rows to be consumed at the clients leisure
func NewRowPull2(ctx context.Context, recordSeq iter.Seq2[arrow.Record, error]) iter.Seq2[Row, error] {
	// Build the push row stream first.
	rows := NewRowSequence(ctx, recordSeq)

	// Enforce single-use to prevent re-iteration after stop/close.
	var used atomic.Bool

	return func(yield func(Row, error) bool) {
		if !used.CompareAndSwap(false, true) {
			return
		}

		// Convert push -> pull using the iter idiom.
		next, stop := iter.Pull2(rows)
		defer stop()

		for {
			row, err, ok := next()
			if !ok {
				return
			}

			// Treat io.EOF as clean termination (don’t forward).
			if errors.Is(err, io.EOF) {
				return
			}
			if err != nil {
				_ = yield(nil, err)
				return
			}
			if !yield(row, nil) {
				return
			}
		}
	}
}
