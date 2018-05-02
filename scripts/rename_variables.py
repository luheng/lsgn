"""From: batzner/tensorflow_rename_variables.py
"""

import sys, getopt
import tensorflow as tf

usage_str = 'python tensorflow_rename_variables.py --checkpoint_dir=path/to/dir/ ' \
            '--replace_from=substr --replace_to=substr --add_prefix=abc --dry_run'

#replacements = {
#  "lm_aggregation/lm_scaling": "module/aggregation/scaling",
#  "lm_aggregation/lm_scores": "module/aggregation/weights"
#}
replacements = {
  "module/aggregation/scaling": "lm_aggregation/lm_scaling",
  "module/aggregation/weights": "lm_aggregation/lm_scores"
}

def rename(checkpoint_dir, replace_from, replace_to, add_prefix, dry_run):
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
            # Set the new name
            new_name = var_name
            for replace_from, replace_to in replacements.iteritems():
                new_name = new_name.replace(replace_from, replace_to)
            
            if dry_run and var_name != new_name:
                print('%s would be renamed to %s.' % (var_name, new_name))
            if not dry_run:
                if var_name != new_name:
                  print('Renaming %s to %s.' % (var_name, new_name))
                  # Copy.
                  var2 = tf.Variable(var, name=var_name)
                else:
                  print('Keeping name %s.' % (var_name))
                # Rename the variable
                var = tf.Variable(var, name=new_name)
        if not dry_run:
            # Save the variables
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.save(sess, checkpoint_dir)  #checkpoint.model_checkpoint_path)


def main(argv):
    checkpoint_dir = None
    replace_from = None
    replace_to = None
    add_prefix = None
    dry_run = False

    try:
        opts, args = getopt.getopt(argv, 'h', ['help=', 'checkpoint_dir=', 'replace_from=',
                                               'replace_to=', 'add_prefix=', 'dry_run'])
    except getopt.GetoptError:
        print(usage_str)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(usage_str)
            sys.exit()
        elif opt == '--checkpoint_dir':
            checkpoint_dir = arg
        elif opt == '--replace_from':
            replace_from = arg
        elif opt == '--replace_to':
            replace_to = arg
        elif opt == '--add_prefix':
            add_prefix = arg
        elif opt == '--dry_run':
            dry_run = True

    if not checkpoint_dir:
        print('Please specify a checkpoint_dir. Usage:')
        print(usage_str)
        sys.exit(2)

    rename(checkpoint_dir, replace_from, replace_to, add_prefix, dry_run)


if __name__ == '__main__':
    main(sys.argv[1:])
