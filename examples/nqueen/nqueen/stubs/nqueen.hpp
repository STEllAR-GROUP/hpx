/*
 * nqueen.hpp
 *      Author: vamatya
 */


#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>

#include <examples/nqueen/nqueen/server/nqueen.hpp>

namespace hpx { namespace components { namespace stubs
{

	struct Board : stub_base<server::Board>
	{

		static void initBoard(naming::id_type gid,unsigned int size, int level)
		{
			applier::apply<server::Board::init_action>(gid, size, level);
		}

		static lcos::future_value<list_t> accessBoard_async(naming::id_type gid)
		{
			typedef server::Board::access_action action_type;
			return lcos::eager_future<action_type>(gid);
		}

		static list_t accessBoard(naming::id_type gid)
		{
			return accessBoard_async(gid).get();
		}

		static void updateBoard(naming::id_type gid, int level, int pos )
		{
			applier::apply<server::Board::update_action>(gid, level, pos);
		}

		static void solveNqueen(naming::id_type gid, Board board, unsigned int size, int level)
		{
			applier::apply<server::Board::solve_action>(gid, board, size, level);
		}

		static lcos::future_value<bool> checkBoard_async(naming::id_type gid, list_t list, int level)
		{
			typedef server::Board::check_action action_type;
			return lcos::eager_future<action_type>(gid, list, level);
		}

		static bool checkBoard(naming::id_type gid, list_t list, int level )
		{
			return checkBoard_async(gid, list, level).get();
		}

	};
}}}
