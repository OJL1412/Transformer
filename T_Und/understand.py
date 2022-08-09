"""

(9,256) => embedding => (9, 256, 512)

(batch_size, -1, 8, 64)
(batch_size, 8, -1, 64)
(batch_size, 8, 64, -1)
(batch_size, 8, -1, -1)
(batch_size, 8, -1, 64)
(batch_size, 8, -1, 64)
(batch_size, -1, 8, 64)

"""