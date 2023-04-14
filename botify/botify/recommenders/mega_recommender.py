from .contextual import Contextual
from .indexed import Indexed
from .recommender import Recommender

# 1000 episodes
# 0.35: 70.3 / 82.7 / 57.9
# 0.62: 69.96 / 82.7
# 0.75: 57 / 70
# 0.45: 58.164 / 69.942 / 46.386
# 0.55: 64.844 / 77.006 / 52.682
# 0.15:	51.49 / 62.849 / 40.132
# 0.65: 68.310 / 80.641 / 55.980
SOLUTION_TIME = 0.62


class MegaRecommender(Recommender):
    """
    Recommend tracks closest to the previous liked one.
    Fall back to the random recommender if no
    recommendations found for the track.
    """

    def __init__(self, tracks_redis, recommendations_redis, user_recommendations_redis, prev_good_redis, catalog):
        self.tracks_redis = tracks_redis
        self.prev_good_redis = prev_good_redis
        self.user_recommendations_redis = user_recommendations_redis
        self.index_fallback = Indexed(tracks_redis, recommendations_redis, catalog)
        self.context_fallback = Contextual(tracks_redis, catalog)
        self.catalog = catalog

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        if prev_track_time <= SOLUTION_TIME:
            if self.prev_good_redis.scard(user) == 0:
                return self.index_fallback.recommend_next(user, prev_track, prev_track_time)

            prev_track = self.prev_good_redis.srandmember(user)

            self.prev_good_redis.srem(user, prev_track)
            self.user_recommendations_redis.srem(prev_track, user)
        else:
            self.prev_good_redis.sadd(user, prev_track)
            self.user_recommendations_redis.sadd(prev_track, user)

        if self.user_recommendations_redis.scard(prev_track) == 0:
            return self.context_fallback.recommend_next(user, prev_track, prev_track_time)

        user_l = self.user_recommendations_redis.srandmember(prev_track)
        return self.context_fallback.recommend_next(user_l, prev_track, prev_track_time)

